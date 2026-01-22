from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
class ForemanAnsibleModule(AnsibleModule):
    """ Baseclass for all foreman related Ansible modules.
        It handles connection parameters and adds the concept of the `foreman_spec`.
        This adds automatic entities resolution based on provided attributes/ sub entities options.

        It adds the following options to foreman_spec 'entity' and 'entity_list' types:

        * search_by (str): Field used to search the sub entity. Defaults to 'name' unless `parent` was set, in which case it defaults to `title`.
        * search_operator (str): Operator used to search the sub entity. Defaults to '='. For fuzzy search use '~'.
        * resource_type (str): Resource type used to build API resource PATH. Defaults to pluralized entity key.
        * resolve (boolean): Defaults to 'True'. If set to false, the sub entity will not be resolved automatically
        * ensure (boolean): Defaults to 'True'. If set to false, it will be removed before sending data to the foreman server.
    """

    def __init__(self, **kwargs):
        self._changed = False
        self._before = defaultdict(list)
        self._after = defaultdict(list)
        self._after_full = defaultdict(list)
        self.foreman_spec, gen_args = _foreman_spec_helper(kwargs.pop('foreman_spec', {}))
        argument_spec = dict(server_url=dict(required=True, fallback=(env_fallback, ['FOREMAN_SERVER_URL', 'FOREMAN_SERVER', 'FOREMAN_URL'])), username=dict(required=True, fallback=(env_fallback, ['FOREMAN_USERNAME', 'FOREMAN_USER'])), password=dict(required=True, no_log=True, fallback=(env_fallback, ['FOREMAN_PASSWORD'])), validate_certs=dict(type='bool', default=True, fallback=(env_fallback, ['FOREMAN_VALIDATE_CERTS'])))
        argument_spec.update(gen_args)
        argument_spec.update(kwargs.pop('argument_spec', {}))
        supports_check_mode = kwargs.pop('supports_check_mode', True)
        self.required_plugins = kwargs.pop('required_plugins', [])
        super(ForemanAnsibleModule, self).__init__(argument_spec=argument_spec, supports_check_mode=supports_check_mode, **kwargs)
        aliases = {alias for arg in argument_spec.values() for alias in arg.get('aliases', [])}
        self.foreman_params = _recursive_dict_without_none(self.params, aliases)
        self.check_requirements()
        self._foremanapi_server_url = self.foreman_params.pop('server_url')
        self._foremanapi_username = self.foreman_params.pop('username')
        self._foremanapi_password = self.foreman_params.pop('password')
        self._foremanapi_validate_certs = self.foreman_params.pop('validate_certs')
        if self._foremanapi_server_url.lower().startswith('http://'):
            self.warn('You have configured a plain HTTP server URL. All communication will happen unencrypted.')
        elif not self._foremanapi_server_url.lower().startswith('https://'):
            self.fail_json(msg='The server URL needs to be either HTTPS or HTTP!')
        self.task_timeout = 60
        self.task_poll = 4
        self._thin_default = False
        self.state = 'undefined'

    @contextmanager
    def api_connection(self):
        """
        Execute a given code block after connecting to the API.

        When the block has finished, call :func:`exit_json` to report that the module has finished to Ansible.
        """
        self.connect()
        yield
        self.exit_json()

    @property
    def changed(self):
        return self._changed

    def set_changed(self):
        self._changed = True

    def _patch_host_update(self):
        _host_methods = self.foremanapi.apidoc['docs']['resources']['hosts']['methods']
        _host_update = next((x for x in _host_methods if x['name'] == 'update'))
        for param in ['location_id', 'organization_id']:
            _host_update_taxonomy_param = next((x for x in _host_update['params'] if x['name'] == param), None)
            if _host_update_taxonomy_param is not None:
                _host_update['params'].remove(_host_update_taxonomy_param)

    @_check_patch_needed(fixed_version='2.2.0', plugins=['remote_execution'])
    def _patch_subnet_rex_api(self):
        """
        This is a workaround for the broken subnet apidoc in foreman remote execution.
        See https://projects.theforeman.org/issues/19086 and https://projects.theforeman.org/issues/30651
        """
        _subnet_rex_proxies_parameter = {u'validations': [], u'name': u'remote_execution_proxy_ids', u'show': True, u'description': u'\n<p>Remote Execution Proxy IDs</p>\n', u'required': False, u'allow_nil': True, u'allow_blank': False, u'full_name': u'subnet[remote_execution_proxy_ids]', u'expected_type': u'array', u'metadata': None, u'validator': u''}
        _subnet_methods = self.foremanapi.apidoc['docs']['resources']['subnets']['methods']
        _subnet_create = next((x for x in _subnet_methods if x['name'] == 'create'))
        _subnet_create_params_subnet = next((x for x in _subnet_create['params'] if x['name'] == 'subnet'))
        _subnet_create_params_subnet['params'].append(_subnet_rex_proxies_parameter)
        _subnet_update = next((x for x in _subnet_methods if x['name'] == 'update'))
        _subnet_update_params_subnet = next((x for x in _subnet_update['params'] if x['name'] == 'subnet'))
        _subnet_update_params_subnet['params'].append(_subnet_rex_proxies_parameter)

    @_check_patch_needed(introduced_version='2.1.0', fixed_version='2.3.0')
    def _patch_subnet_externalipam_group_api(self):
        """
        This is a workaround for the broken subnet apidoc for External IPAM.
        See https://projects.theforeman.org/issues/30890
        """
        _subnet_externalipam_group_parameter = {u'validations': [], u'name': u'externalipam_group', u'show': True, u'description': u'\n<p>External IPAM group - only relevant when IPAM is set to external</p>\n', u'required': False, u'allow_nil': True, u'allow_blank': False, u'full_name': u'subnet[externalipam_group]', u'expected_type': u'string', u'metadata': None, u'validator': u''}
        _subnet_methods = self.foremanapi.apidoc['docs']['resources']['subnets']['methods']
        _subnet_create = next((x for x in _subnet_methods if x['name'] == 'create'))
        _subnet_create_params_subnet = next((x for x in _subnet_create['params'] if x['name'] == 'subnet'))
        _subnet_create_params_subnet['params'].append(_subnet_externalipam_group_parameter)
        _subnet_update = next((x for x in _subnet_methods if x['name'] == 'update'))
        _subnet_update_params_subnet = next((x for x in _subnet_update['params'] if x['name'] == 'subnet'))
        _subnet_update_params_subnet['params'].append(_subnet_externalipam_group_parameter)

    @_check_patch_needed(plugins=['katello'])
    def _patch_organization_update_api(self):
        """
        This is a workaround for the broken organization update apidoc in Katello.
        See https://projects.theforeman.org/issues/27538
        """
        _organization_methods = self.foremanapi.apidoc['docs']['resources']['organizations']['methods']
        _organization_update = next((x for x in _organization_methods if x['name'] == 'update'))
        _organization_update_params_organization = next((x for x in _organization_update['params'] if x['name'] == 'organization'))
        _organization_update_params_organization['required'] = False

    @_check_patch_needed(plugins=['katello'])
    def _patch_cv_filter_rule_api(self):
        """
        This is a workaround for missing params of CV Filter Rule update controller in Katello.
        See https://projects.theforeman.org/issues/30908
        """
        _content_view_filter_rule_methods = self.foremanapi.apidoc['docs']['resources']['content_view_filter_rules']['methods']
        _content_view_filter_rule_create = next((x for x in _content_view_filter_rule_methods if x['name'] == 'create'))
        _content_view_filter_rule_update = next((x for x in _content_view_filter_rule_methods if x['name'] == 'update'))
        for param_name in ['uuid', 'errata_ids', 'date_type', 'module_stream_ids']:
            create_param = next((x for x in _content_view_filter_rule_create['params'] if x['name'] == param_name), None)
            update_param = next((x for x in _content_view_filter_rule_update['params'] if x['name'] == param_name), None)
            if create_param is not None and update_param is None:
                _content_view_filter_rule_update['params'].append(create_param)

    @_check_patch_needed(fixed_version='3.5.0', plugins=['katello'])
    def _patch_ak_product_content_per_page(self):
        """
        This is a workaround for the API not exposing the per_page param on the product_content endpoint
        See https://projects.theforeman.org/issues/35633
        """
        _per_page_param = {'name': 'per_page', 'full_name': 'per_page', 'description': '\n<p>Number of results per page to return</p>\n', 'required': False, 'allow_nil': False, 'allow_blank': False, 'validator': 'Must be a number.', 'expected_type': 'numeric', 'metadata': None, 'show': True, 'validations': []}
        _ak_methods = self.foremanapi.apidoc['docs']['resources']['activation_keys']['methods']
        _ak_product_content = next((x for x in _ak_methods if x['name'] == 'product_content'))
        if next((x for x in _ak_product_content['params'] if x['name'] == 'per_page'), None) is None:
            _ak_product_content['params'].append(_per_page_param)

    @_check_patch_needed(fixed_version='3.5.0', plugins=['katello'])
    def _patch_organization_ignore_types_api(self):
        """
        This is a workaround for the missing ignore_types in the organization apidoc in Katello.
        See https://projects.theforeman.org/issues/35687
        """
        _ignore_types_param = {'name': 'ignore_types', 'full_name': 'organization[ignore_types]', 'description': '\n<p>List of resources types that will be automatically associated</p>\n', 'required': False, 'allow_nil': True, 'allow_blank': False, 'validator': 'Must be an array of any type', 'expected_type': 'array', 'metadata': None, 'show': True, 'validations': []}
        _organization_methods = self.foremanapi.apidoc['docs']['resources']['organizations']['methods']
        _organization_create = next((x for x in _organization_methods if x['name'] == 'create'))
        _organization_update = next((x for x in _organization_methods if x['name'] == 'update'))
        if next((x for x in _organization_create['params'] if x['name'] == 'ignore_types'), None) is None:
            _organization_create['params'].append(_ignore_types_param)
            _organization_update['params'].append(_ignore_types_param)

    @_check_patch_needed(fixed_version='3.8.0', plugins=['katello'])
    def _patch_products_repositories_allow_nil_credential(self):
        """
        This is a workaround for the missing allow_nil: true in the Products and Repositories controllers
        See https://projects.theforeman.org/issues/36497
        """
        for resource in ['products', 'repositories']:
            methods = self.foremanapi.apidoc['docs']['resources'][resource]['methods']
            for action in ['create', 'update']:
                resource_action = next((x for x in methods if x['name'] == action))
                for param in ['gpg_key_id', 'ssl_ca_cert_id', 'ssl_client_cert_id', 'ssl_client_key_id']:
                    resource_param = next((x for x in resource_action['params'] if x['name'] == param))
                    resource_param['allow_nil'] = True

    def check_requirements(self):
        if not HAS_APYPIE:
            self.fail_json(msg=missing_required_lib('requests'), exception=APYPIE_IMP_ERR)

    @_exception2fail_json(msg='Failed to connect to Foreman server: {0}')
    def connect(self):
        """
        Connect to the Foreman API.

        This will create a new ``apypie.Api`` instance using the provided server information,
        check that the API is actually reachable (by calling :func:`status`),
        apply any required patches to the apidoc and ensure the server has all the plugins installed
        that are required by the module.
        """
        self.foremanapi = apypie.Api(uri=self._foremanapi_server_url, username=to_bytes(self._foremanapi_username), password=to_bytes(self._foremanapi_password), api_version=2, verify_ssl=self._foremanapi_validate_certs)
        _status = self.status()
        self.foreman_version = LooseVersion(_status.get('version', '0.0.0'))
        self.apply_apidoc_patches()
        self.check_required_plugins()

    def apply_apidoc_patches(self):
        """
        Apply patches to the local apidoc representation.
        When adding another patch, consider that the endpoint in question may depend on a plugin to be available.
        If possible, make the patch only execute on specific server/plugin versions.
        """
        self._patch_host_update()
        self._patch_subnet_rex_api()
        self._patch_subnet_externalipam_group_api()
        self._patch_organization_update_api()
        self._patch_cv_filter_rule_api()
        self._patch_ak_product_content_per_page()
        self._patch_organization_ignore_types_api()
        self._patch_products_repositories_allow_nil_credential()

    @_exception2fail_json(msg='Failed to connect to Foreman server: {0}')
    def status(self):
        """
        Call the ``status`` API endpoint to ensure the server is reachable.

        :return: The full API response
        :rtype: dict
        """
        return self.foremanapi.resource('home').call('status')

    def _resource(self, resource):
        if resource not in self.foremanapi.resources:
            raise Exception("The server doesn't know about {0}, is the right plugin installed?".format(resource))
        return self.foremanapi.resource(resource)

    def _resource_call(self, resource, *args, **kwargs):
        return self._resource(resource).call(*args, **kwargs)

    def _resource_prepare_params(self, resource, action, params):
        api_action = self._resource(resource).action(action)
        return api_action.prepare_params(params)

    @_exception2fail_json(msg='Failed to show resource: {0}')
    def show_resource(self, resource, resource_id, params=None):
        """
        Execute the ``show`` action on an entity.

        :param resource: Plural name of the api resource to show
        :type resource: str
        :param resource_id: The ID of the entity to show
        :type resource_id: int
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: Union[dict,None], optional
        """
        if params is None:
            params = {}
        else:
            params = params.copy()
        params['id'] = resource_id
        params = self._resource_prepare_params(resource, 'show', params)
        return self._resource_call(resource, 'show', params)

    @_exception2fail_json(msg='Failed to list resource: {0}')
    def list_resource(self, resource, search=None, params=None):
        """
        Execute the ``index`` action on an resource.

        :param resource: Plural name of the api resource to show
        :type resource: str
        :param search: Search string as accepted by the API to limit the results
        :type search: str, optional
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: Union[dict,None], optional
        """
        if params is None:
            params = {}
        else:
            params = params.copy()
        if search is not None:
            params['search'] = search
        params['per_page'] = PER_PAGE
        params = self._resource_prepare_params(resource, 'index', params)
        return self._resource_call(resource, 'index', params)['results']

    def find_resource(self, resource, search, params=None, failsafe=False, thin=None):
        list_params = {}
        if params is not None:
            list_params.update(params)
        if thin is None:
            thin = self._thin_default
        list_params['thin'] = thin
        results = self.list_resource(resource, search, list_params)
        if len(results) == 1:
            result = results[0]
        elif failsafe:
            result = None
        else:
            if len(results) > 1:
                error_msg = 'too many ({0})'.format(len(results))
            else:
                error_msg = 'no'
            self.fail_json(msg='Found {0} results while searching for {1} with {2}'.format(error_msg, resource, search))
        if result and (not thin):
            result = self.show_resource(resource, result['id'], params=params)
        return result

    def find_resource_by(self, resource, search_field, value, **kwargs):
        if not value:
            return NoEntity
        search = '{0}{1}"{2}"'.format(search_field, kwargs.pop('search_operator', '='), value)
        return self.find_resource(resource, search, **kwargs)

    def find_resource_by_name(self, resource, name, **kwargs):
        return self.find_resource_by(resource, 'name', name, **kwargs)

    def find_resource_by_title(self, resource, title, **kwargs):
        return self.find_resource_by(resource, 'title', title, **kwargs)

    def find_resource_by_id(self, resource, obj_id, **kwargs):
        return self.find_resource_by(resource, 'id', obj_id, **kwargs)

    def find_resources_by_name(self, resource, names, **kwargs):
        return [self.find_resource_by_name(resource, name, **kwargs) for name in names]

    def find_operatingsystem(self, name, failsafe=False, **kwargs):
        result = self.find_resource_by_title('operatingsystems', name, failsafe=True, **kwargs)
        if not result:
            result = self.find_resource_by('operatingsystems', 'title', name, search_operator='~', failsafe=failsafe, **kwargs)
        return result

    def find_puppetclass(self, name, environment=None, params=None, failsafe=False, thin=None):
        if thin is None:
            thin = self._thin_default
        if environment:
            scope = {'environment_id': environment}
        else:
            scope = {}
        if params is not None:
            scope.update(params)
        search = 'name="{0}"'.format(name)
        results = self.list_resource('puppetclasses', search, params=scope)
        if len(results) == 1 and len(list(results.values())[0]) == 1:
            result = list(results.values())[0][0]
            if thin:
                return {'id': result['id']}
            else:
                return result
        if failsafe:
            return None
        else:
            self.fail_json(msg='No data found for name="%s"' % search)

    def find_puppetclasses(self, names, **kwargs):
        return [self.find_puppetclass(name, **kwargs) for name in names]

    def find_cluster(self, name, compute_resource):
        cluster = self.find_compute_resource_parts('clusters', name, compute_resource, None, ['ovirt', 'vmware'])
        if compute_resource['provider'].lower() == 'vmware':
            path_or_name = cluster.get('full_path', cluster['name'])
            cluster['_api_identifier'] = path_or_name
        else:
            cluster['_api_identifier'] = cluster['id']
        return cluster

    def find_network(self, name, compute_resource, cluster=None):
        return self.find_compute_resource_parts('networks', name, compute_resource, cluster, ['ovirt', 'vmware', 'google', 'azurerm'])

    def find_storage_domain(self, name, compute_resource, cluster=None):
        return self.find_compute_resource_parts('storage_domains', name, compute_resource, cluster, ['ovirt', 'vmware'])

    def find_storage_pod(self, name, compute_resource, cluster=None):
        return self.find_compute_resource_parts('storage_pods', name, compute_resource, cluster, ['vmware'])

    def find_compute_resource_parts(self, part_name, name, compute_resource, cluster=None, supported_crs=None):
        if supported_crs is None:
            supported_crs = []
        if compute_resource['provider'].lower() not in supported_crs:
            return {'id': name, 'name': name}
        additional_params = {'id': compute_resource['id']}
        if cluster is not None:
            additional_params['cluster_id'] = six.moves.urllib.parse.quote(cluster['_api_identifier'], safe='')
        api_name = 'available_{0}'.format(part_name)
        available_parts = self.resource_action('compute_resources', api_name, params=additional_params, ignore_check_mode=True, record_change=False)['results']
        part = next((part for part in available_parts if str(part['name']) == str(name) or str(part['id']) == str(name) or part.get('full_path') == str(name)), None)
        if part is None:
            err_msg = "Could not find {0} '{1}' on compute resource '{2}'.".format(part_name, name, compute_resource.get('name'))
            self.fail_json(msg=err_msg)
        return part

    def scope_for(self, key, scoped_resource=None):
        if scoped_resource in ['content_views', 'repositories'] and key == 'lifecycle_environment':
            scope_key = 'environment'
        else:
            scope_key = key
        return {'{0}_id'.format(scope_key): self.lookup_entity(key)['id']}

    def set_entity(self, key, entity):
        self.foreman_params[key] = entity

    def lookup_entity(self, key, params=None):
        if key not in self.foreman_params:
            return None
        entity_spec = self.foreman_spec[key]
        if _is_resolved(entity_spec, self.foreman_params[key]):
            return self.foreman_params[key]
        result = self._lookup_entity(self.foreman_params[key], entity_spec, params)
        self.set_entity(key, result)
        return result

    def _lookup_entity(self, identifier, entity_spec, params=None):
        if identifier is NoEntity:
            return NoEntity
        resource_type = entity_spec['resource_type']
        failsafe = entity_spec.get('failsafe', False)
        thin = entity_spec.get('thin', True)
        if params:
            params = params.copy()
        else:
            params = {}
        try:
            for scope in entity_spec.get('scope', []):
                params.update(self.scope_for(scope, resource_type))
            for optional_scope in entity_spec.get('optional_scope', []):
                if optional_scope in self.foreman_params:
                    params.update(self.scope_for(optional_scope, resource_type))
        except TypeError:
            if failsafe:
                if entity_spec.get('type') == 'entity':
                    result = None
                else:
                    result = [None for value in identifier]
            else:
                self.fail_json(msg='Failed to lookup scope {0} while searching for {1}.'.format(entity_spec['scope'], resource_type))
        else:
            if resource_type == 'operatingsystems':
                if entity_spec.get('type') == 'entity':
                    result = self.find_operatingsystem(identifier, params=params, failsafe=failsafe, thin=thin)
                else:
                    result = [self.find_operatingsystem(value, params=params, failsafe=failsafe, thin=thin) for value in identifier]
            elif resource_type == 'puppetclasses':
                if entity_spec.get('type') == 'entity':
                    result = self.find_puppetclass(identifier, params=params, failsafe=failsafe, thin=thin)
                else:
                    result = [self.find_puppetclass(value, params=params, failsafe=failsafe, thin=thin) for value in identifier]
            elif entity_spec.get('type') == 'entity':
                result = self.find_resource_by(resource=resource_type, value=identifier, search_field=entity_spec.get('search_by', ENTITY_KEYS.get(resource_type, 'name')), search_operator=entity_spec.get('search_operator', '='), failsafe=failsafe, thin=thin, params=params)
            else:
                result = [self.find_resource_by(resource=resource_type, value=value, search_field=entity_spec.get('search_by', ENTITY_KEYS.get(resource_type, 'name')), search_operator=entity_spec.get('search_operator', '='), failsafe=failsafe, thin=thin, params=params) for value in identifier]
        return result

    def auto_lookup_entities(self):
        self.auto_lookup_nested_entities()
        return [self.lookup_entity(key) for key, entity_spec in self.foreman_spec.items() if entity_spec.get('resolve', True) and entity_spec.get('type') in {'entity', 'entity_list'}]

    def auto_lookup_nested_entities(self):
        for key, entity_spec in self.foreman_spec.items():
            if entity_spec.get('type') in {'nested_list'}:
                for nested_key, nested_spec in entity_spec['foreman_spec'].items():
                    for item in self.foreman_params.get(key, []):
                        if nested_key in item and nested_spec.get('resolve', True) and (not _is_resolved(nested_spec, item[nested_key])):
                            item[nested_key] = self._lookup_entity(item[nested_key], nested_spec)

    def record_before(self, resource, entity):
        if isinstance(entity, dict):
            to_record = _recursive_dict_without_none(entity)
        else:
            to_record = entity
        self._before[resource].append(to_record)

    def record_after(self, resource, entity):
        if isinstance(entity, dict):
            to_record = _recursive_dict_without_none(entity)
        else:
            to_record = entity
        self._after[resource].append(to_record)

    def record_after_full(self, resource, entity):
        self._after_full[resource].append(entity)

    @_exception2fail_json(msg='Failed to ensure entity state: {0}')
    def ensure_entity(self, resource, desired_entity, current_entity, params=None, state=None, foreman_spec=None):
        """
        Ensure that a given entity has a certain state

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param desired_entity: Desired properties of the entity
        :type desired_entity: dict
        :param current_entity: Current properties of the entity or None if nonexistent
        :type current_entity: Union[dict,None]
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional
        :param state: Desired state of the entity (optionally taken from the module)
        :type state: str, optional
        :param foreman_spec: Description of the entity structure (optionally taken from module)
        :type foreman_spec: dict, optional

        :return: The new current state of the entity
        :rtype: Union[dict,None]
        """
        if state is None:
            state = self.state
        if foreman_spec is None:
            foreman_spec = self.foreman_spec
        else:
            foreman_spec, _dummy = _foreman_spec_helper(foreman_spec)
        updated_entity = None
        self.record_before(resource, _flatten_entity(current_entity, foreman_spec))
        if state == 'present_with_defaults':
            if current_entity is None:
                updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
        elif state == 'present':
            if current_entity is None:
                updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
            else:
                updated_entity = self._update_entity(resource, desired_entity, current_entity, params, foreman_spec)
        elif state == 'copied':
            if current_entity is not None:
                updated_entity = self._copy_entity(resource, desired_entity, current_entity, params)
        elif state == 'reverted':
            if current_entity is not None:
                updated_entity = self._revert_entity(resource, current_entity, params)
        elif state == 'new_snapshot':
            updated_entity = self._create_entity(resource, desired_entity, params, foreman_spec)
        elif state == 'absent':
            if current_entity is not None:
                updated_entity = self._delete_entity(resource, current_entity, params)
        else:
            self.fail_json(msg='Not a valid state: {0}'.format(state))
        self.record_after(resource, _flatten_entity(updated_entity, foreman_spec))
        self.record_after_full(resource, updated_entity)
        return updated_entity

    def _validate_supported_payload(self, resource, action, payload):
        """
        Check whether the payload only contains supported keys.
        Emits a warning for keys that are not part of the apidoc.

        :param resource: Plural name of the api resource to check
        :type resource: str
        :param action: Name of the action to check payload against
        :type action: str
        :param payload: API paylod to be checked
        :type payload: dict

        :return: The payload as it can be submitted to the API
        :rtype: dict
        """
        filtered_payload = self._resource_prepare_params(resource, action, payload)
        unsupported_parameters = set(payload.keys()) - set(_recursive_dict_keys(filtered_payload))
        if unsupported_parameters:
            warn_msg = 'The following parameters are not supported by your server when performing {0} on {1}: {2}. They were ignored.'
            self.warn(warn_msg.format(action, resource, unsupported_parameters))
        return filtered_payload

    def _create_entity(self, resource, desired_entity, params, foreman_spec):
        """
        Create entity with given properties

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param desired_entity: Desired properties of the entity
        :type desired_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional
        :param foreman_spec: Description of the entity structure
        :type foreman_spec: dict

        :return: The new current state of the entity
        :rtype: dict
        """
        payload = _flatten_entity(desired_entity, foreman_spec)
        self._validate_supported_payload(resource, 'create', payload)
        if not self.check_mode:
            if params:
                payload.update(params)
            return self.resource_action(resource, 'create', payload)
        else:
            fake_entity = desired_entity.copy()
            fake_entity['id'] = -1
            self.set_changed()
            return fake_entity

    def _update_entity(self, resource, desired_entity, current_entity, params, foreman_spec):
        """
        Update a given entity with given properties if any diverge

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param desired_entity: Desired properties of the entity
        :type desired_entity: dict
        :param current_entity: Current properties of the entity
        :type current_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional
        :param foreman_spec: Description of the entity structure
        :type foreman_spec: dict

        :return: The new current state of the entity
        :rtype: dict
        """
        payload = {}
        desired_entity = _flatten_entity(desired_entity, foreman_spec)
        current_flat_entity = _flatten_entity(current_entity, foreman_spec)
        for key, value in desired_entity.items():
            foreman_type = foreman_spec[key].get('type', 'str')
            new_value = value
            old_value = current_flat_entity.get(key)
            if foreman_type == 'str':
                old_value = to_native(old_value)
                new_value = to_native(new_value)
            elif foreman_type == 'list' and value and isinstance(value[0], dict):
                if 'name' in value[0]:
                    sort_key = 'name'
                else:
                    sort_key = list(value[0].keys())[0]
                new_value = sorted(new_value, key=operator.itemgetter(sort_key))
                old_value = sorted(old_value, key=operator.itemgetter(sort_key))
            if new_value != old_value:
                payload[key] = value
        if self._validate_supported_payload(resource, 'update', payload):
            payload['id'] = current_flat_entity['id']
            if not self.check_mode:
                if params:
                    payload.update(params)
                return self.resource_action(resource, 'update', payload)
            else:
                fake_entity = current_flat_entity.copy()
                fake_entity.update(payload)
                self.set_changed()
                return fake_entity
        else:
            return current_entity

    def _copy_entity(self, resource, desired_entity, current_entity, params):
        """
        Copy a given entity

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param desired_entity: Desired properties of the entity
        :type desired_entity: dict
        :param current_entity: Current properties of the entity
        :type current_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional

        :return: The new current state of the entity
        :rtype: dict
        """
        payload = {'id': current_entity['id'], 'new_name': desired_entity['new_name']}
        if params:
            payload.update(params)
        return self.resource_action(resource, 'copy', payload)

    def _revert_entity(self, resource, current_entity, params):
        """
        Revert a given entity

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param current_entity: Current properties of the entity
        :type current_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional

        :return: The new current state of the entity
        :rtype: dict
        """
        payload = {'id': current_entity['id']}
        if params:
            payload.update(params)
        return self.resource_action(resource, 'revert', payload)

    def _delete_entity(self, resource, current_entity, params):
        """
        Delete a given entity

        :param resource: Plural name of the api resource to manipulate
        :type resource: str
        :param current_entity: Current properties of the entity
        :type current_entity: dict
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: dict, optional

        :return: The new current state of the entity
        :rtype: Union[dict,None]
        """
        payload = {'id': current_entity['id']}
        if params:
            payload.update(params)
        entity = self.resource_action(resource, 'destroy', payload)
        if entity and isinstance(entity, dict) and ('error' in entity) and ('message' in entity['error']):
            self.fail_json(msg=entity['error']['message'])
        return None

    def resource_action(self, resource, action, params, options=None, data=None, files=None, ignore_check_mode=False, record_change=True, ignore_task_errors=False):
        resource_payload = self._resource_prepare_params(resource, action, params)
        if options is None:
            options = {}
        try:
            result = None
            if ignore_check_mode or not self.check_mode:
                result = self._resource_call(resource, action, resource_payload, options=options, data=data, files=files)
                is_foreman_task = isinstance(result, dict) and 'action' in result and ('state' in result) and ('started_at' in result)
                if is_foreman_task:
                    result = self.wait_for_task(result, ignore_errors=ignore_task_errors)
        except Exception as e:
            msg = 'Error while performing {0} on {1}: {2}'.format(action, resource, to_native(e))
            self.fail_from_exception(e, msg)
        if record_change and (not ignore_check_mode):
            self.set_changed()
        return result

    def wait_for_task(self, task, ignore_errors=False):
        duration = self.task_timeout
        while task['state'] not in ['paused', 'stopped']:
            duration -= self.task_poll
            if duration <= 0:
                self.fail_json(msg='Timeout waiting for Task {0}'.format(task['id']))
            time.sleep(self.task_poll)
            resource_payload = self._resource_prepare_params('foreman_tasks', 'show', {'id': task['id']})
            task = self._resource_call('foreman_tasks', 'show', resource_payload)
        if not ignore_errors and task['result'] != 'success':
            self.fail_json(msg='Task {0}({1}) did not succeed. Task information: {2}'.format(task['action'], task['id'], task['humanized']['errors']))
        return task

    def fail_from_exception(self, exc, msg):
        fail = {'msg': msg}
        if isinstance(exc, requests.exceptions.HTTPError):
            try:
                response = exc.response.json()
                if 'error' in response:
                    fail['error'] = response['error']
                else:
                    fail['error'] = response
            except Exception:
                fail['error'] = exc.response.text
        self.fail_json(**fail)

    def exit_json(self, changed=False, **kwargs):
        kwargs['changed'] = changed or self.changed
        if 'diff' not in kwargs and (self._before or self._after):
            kwargs['diff'] = {'before': self._before, 'after': self._after}
        if 'entity' not in kwargs and self._after_full:
            kwargs['entity'] = self._after_full
        super(ForemanAnsibleModule, self).exit_json(**kwargs)

    def has_plugin(self, plugin_name):
        try:
            resource_name = _PLUGIN_RESOURCES[plugin_name]
        except KeyError:
            raise Exception('Unknown plugin: {0}'.format(plugin_name))
        return resource_name in self.foremanapi.resources

    def check_required_plugins(self):
        missing_plugins = []
        for plugin, params in self.required_plugins:
            for param in params:
                if (param in self.foreman_params or param == '*') and (not self.has_plugin(plugin)):
                    if param == '*':
                        param = 'the whole module'
                    missing_plugins.append('{0} (for {1})'.format(plugin, param))
        if missing_plugins:
            missing_msg = 'The server is missing required plugins: {0}.'.format(', '.join(missing_plugins))
            self.fail_json(msg=missing_msg)