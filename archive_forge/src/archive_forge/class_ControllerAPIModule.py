from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
class ControllerAPIModule(ControllerModule):
    _COLLECTION_VERSION = '23.9.0'
    _COLLECTION_TYPE = 'awx'
    collection_to_version = {'awx': 'AWX', 'controller': 'Red Hat Ansible Automation Platform'}
    session = None
    IDENTITY_FIELDS = {'users': 'username', 'workflow_job_template_nodes': 'identifier', 'instances': 'hostname'}
    ENCRYPTED_STRING = '$encrypted$'

    def __init__(self, argument_spec, direct_params=None, error_callback=None, warn_callback=None, **kwargs):
        kwargs['supports_check_mode'] = True
        super().__init__(argument_spec=argument_spec, direct_params=direct_params, error_callback=error_callback, warn_callback=warn_callback, **kwargs)
        self.session = Request(cookies=CookieJar(), timeout=self.request_timeout, validate_certs=self.verify_ssl)
        if 'update_secrets' in self.params:
            self.update_secrets = self.params.pop('update_secrets')
        else:
            self.update_secrets = True

    @staticmethod
    def get_name_field_from_endpoint(endpoint):
        return ControllerAPIModule.IDENTITY_FIELDS.get(endpoint, 'name')

    def get_item_name(self, item, allow_unknown=False):
        if item:
            if 'name' in item:
                return item['name']
            for field_name in ControllerAPIModule.IDENTITY_FIELDS.values():
                if field_name in item:
                    return item[field_name]
            if item.get('type', None) in ('o_auth2_access_token', 'credential_input_source'):
                return item['id']
        if allow_unknown:
            return 'unknown'
        if item:
            self.exit_json(msg='Cannot determine identity field for {0} object.'.format(item.get('type', 'unknown')))
        else:
            self.exit_json(msg='Cannot determine identity field for Undefined object.')

    def head_endpoint(self, endpoint, *args, **kwargs):
        return self.make_request('HEAD', endpoint, **kwargs)

    def get_endpoint(self, endpoint, *args, **kwargs):
        return self.make_request('GET', endpoint, **kwargs)

    def patch_endpoint(self, endpoint, *args, **kwargs):
        if self.check_mode:
            self.json_output['changed'] = True
            self.exit_json(**self.json_output)
        return self.make_request('PATCH', endpoint, **kwargs)

    def post_endpoint(self, endpoint, *args, **kwargs):
        if self.check_mode:
            self.json_output['changed'] = True
            self.exit_json(**self.json_output)
        return self.make_request('POST', endpoint, **kwargs)

    def delete_endpoint(self, endpoint, *args, **kwargs):
        if self.check_mode:
            self.json_output['changed'] = True
            self.exit_json(**self.json_output)
        return self.make_request('DELETE', endpoint, **kwargs)

    def get_all_endpoint(self, endpoint, *args, **kwargs):
        response = self.get_endpoint(endpoint, *args, **kwargs)
        if 'next' not in response['json']:
            raise RuntimeError('Expected list from API at {0}, got: {1}'.format(endpoint, response))
        next_page = response['json']['next']
        if response['json']['count'] > 10000:
            self.fail_json(msg='The number of items being queried for is higher than 10,000.')
        while next_page is not None:
            next_response = self.get_endpoint(next_page)
            response['json']['results'] = response['json']['results'] + next_response['json']['results']
            next_page = next_response['json']['next']
            response['json']['next'] = next_page
        return response

    def get_one(self, endpoint, name_or_id=None, allow_none=True, check_exists=False, **kwargs):
        new_kwargs = kwargs.copy()
        response = None
        if name_or_id is not None and '++' in name_or_id:
            url_quoted_name = quote(name_or_id, safe='+')
            named_endpoint = '{0}/{1}/'.format(endpoint, url_quoted_name)
            named_response = self.get_endpoint(named_endpoint)
            if named_response['status_code'] == 200 and 'json' in named_response:
                response = {'json': {'count': 1, 'results': [named_response['json']]}}
        if response is None:
            if name_or_id:
                name_field = self.get_name_field_from_endpoint(endpoint)
                new_data = kwargs.get('data', {}).copy()
                if name_field in new_data:
                    self.fail_json(msg="You can't specify the field {0} in your search data if using the name_or_id field".format(name_field))
                try:
                    new_data['or__id'] = int(name_or_id)
                    new_data['or__{0}'.format(name_field)] = name_or_id
                except ValueError:
                    new_data[name_field] = name_or_id
                new_kwargs['data'] = new_data
            response = self.get_endpoint(endpoint, **new_kwargs)
            if response['status_code'] != 200:
                fail_msg = 'Got a {0} response when trying to get one from {1}'.format(response['status_code'], endpoint)
                if 'detail' in response.get('json', {}):
                    fail_msg += ', detail: {0}'.format(response['json']['detail'])
                self.fail_json(msg=fail_msg)
            if 'count' not in response['json'] or 'results' not in response['json']:
                self.fail_json(msg='The endpoint did not provide count and results')
        if response['json']['count'] == 0:
            if allow_none:
                return None
            else:
                self.fail_wanted_one(response, endpoint, new_kwargs.get('data'))
        elif response['json']['count'] > 1:
            if name_or_id:
                for asset in response['json']['results']:
                    if str(asset['id']) == name_or_id:
                        return asset
            self.fail_wanted_one(response, endpoint, new_kwargs.get('data'))
        if check_exists:
            self.json_output['id'] = response['json']['results'][0]['id']
            self.exit_json(**self.json_output)
        return response['json']['results'][0]

    def fail_wanted_one(self, response, endpoint, query_params):
        sample = response.copy()
        if len(sample['json']['results']) > 1:
            sample['json']['results'] = sample['json']['results'][:2] + ['...more results snipped...']
        url = self.build_url(endpoint, query_params)
        host_length = len(self.host)
        display_endpoint = url.geturl()[host_length:]
        self.fail_json(msg='Request to {0} returned {1} items, expected 1'.format(display_endpoint, response['json']['count']), query=query_params, response=sample, total_results=response['json']['count'])

    def get_exactly_one(self, endpoint, name_or_id=None, **kwargs):
        return self.get_one(endpoint, name_or_id=name_or_id, allow_none=False, **kwargs)

    def resolve_name_to_id(self, endpoint, name_or_id):
        return self.get_exactly_one(endpoint, name_or_id)['id']

    def make_request(self, method, endpoint, *args, **kwargs):
        if not method:
            raise Exception('The HTTP method must be defined')
        if method in ['POST', 'PUT', 'PATCH']:
            url = self.build_url(endpoint)
        else:
            url = self.build_url(endpoint, query_params=kwargs.get('data'))
        headers = kwargs.get('headers', {})
        if not self.oauth_token and (not self.authenticated):
            self.authenticate(**kwargs)
        if self.oauth_token:
            headers['Authorization'] = 'Bearer {0}'.format(self.oauth_token)
        if method in ['POST', 'PUT', 'PATCH']:
            headers.setdefault('Content-Type', 'application/json')
            kwargs['headers'] = headers
        data = None
        if headers.get('Content-Type', '') == 'application/json':
            data = dumps(kwargs.get('data', {}))
        try:
            response = self.session.open(method, url.geturl(), headers=headers, timeout=self.request_timeout, validate_certs=self.verify_ssl, follow_redirects=True, data=data)
        except SSLValidationError as ssl_err:
            self.fail_json(msg='Could not establish a secure connection to your host ({1}): {0}.'.format(url.netloc, ssl_err))
        except ConnectionError as con_err:
            self.fail_json(msg='There was a network error of some kind trying to connect to your host ({1}): {0}.'.format(url.netloc, con_err))
        except HTTPError as he:
            if he.code >= 500:
                self.fail_json(msg='The host sent back a server error ({1}): {0}. Please check the logs and try again later'.format(url.path, he))
            elif he.code == 401:
                self.fail_json(msg='Invalid authentication credentials for {0} (HTTP 401).'.format(url.path))
            elif he.code == 403:
                self.fail_json(msg="You don't have permission to {1} to {0} (HTTP 403).".format(url.path, method))
            elif he.code == 404:
                if kwargs.get('return_none_on_404', False):
                    return None
                self.fail_json(msg='The requested object could not be found at {0}.'.format(url.path))
            elif he.code == 405:
                self.fail_json(msg='Cannot make a request with the {0} method to this endpoint {1}'.format(method, url.path))
            elif he.code >= 400:
                page_data = he.read()
                try:
                    return {'status_code': he.code, 'json': loads(page_data)}
                except ValueError:
                    return {'status_code': he.code, 'text': page_data}
            elif he.code == 204 and method == 'DELETE':
                pass
            else:
                self.fail_json(msg='Unexpected return code when calling {0}: {1}'.format(url.geturl(), he))
        except Exception as e:
            self.fail_json(msg='There was an unknown error when trying to connect to {2}: {0} {1}'.format(type(e).__name__, e, url.geturl()))
        if not self.version_checked:
            try:
                controller_type = response.getheader('X-API-Product-Name', None)
                controller_version = response.getheader('X-API-Product-Version', None)
            except Exception:
                controller_type = response.info().getheader('X-API-Product-Name', None)
                controller_version = response.info().getheader('X-API-Product-Version', None)
            parsed_collection_version = Version(self._COLLECTION_VERSION).version
            if controller_version:
                parsed_controller_version = Version(controller_version).version
                if controller_type == 'AWX':
                    collection_compare_ver = parsed_collection_version[0]
                    controller_compare_ver = parsed_controller_version[0]
                else:
                    collection_compare_ver = '{0}.{1}'.format(parsed_collection_version[0], parsed_collection_version[1])
                    controller_compare_ver = '{0}.{1}'.format(parsed_controller_version[0], parsed_controller_version[1])
                if self._COLLECTION_TYPE not in self.collection_to_version or self.collection_to_version[self._COLLECTION_TYPE] != controller_type:
                    self.warn('You are using the {0} version of this collection but connecting to {1}'.format(self._COLLECTION_TYPE, controller_type))
                elif collection_compare_ver != controller_compare_ver:
                    self.warn('You are running collection version {0} but connecting to {2} version {1}'.format(self._COLLECTION_VERSION, controller_version, controller_type))
            self.version_checked = True
        response_body = ''
        try:
            response_body = response.read()
        except Exception as e:
            self.fail_json(msg='Failed to read response body: {0}'.format(e))
        response_json = {}
        if response_body and response_body != '':
            try:
                response_json = loads(response_body)
            except Exception as e:
                self.fail_json(msg='Failed to parse the response json: {0}'.format(e))
        if PY2:
            status_code = response.getcode()
        else:
            status_code = response.status
        return {'status_code': status_code, 'json': response_json}

    def authenticate(self, **kwargs):
        if self.username and self.password:
            login_data = {'description': 'Automation Platform Controller Module Token', 'application': None, 'scope': 'write'}
            endpoint = self.url_prefix.rstrip('/') + '/api/v2/tokens/'
            api_token_url = self.url._replace(path=endpoint).geturl()
            try:
                response = self.session.open('POST', api_token_url, validate_certs=self.verify_ssl, timeout=self.request_timeout, follow_redirects=True, force_basic_auth=True, url_username=self.username, url_password=self.password, data=dumps(login_data), headers={'Content-Type': 'application/json'})
            except HTTPError as he:
                try:
                    resp = he.read()
                except Exception as e:
                    resp = 'unknown {0}'.format(e)
                self.fail_json(msg='Failed to get token: {0}'.format(he), response=resp)
            except Exception as e:
                self.fail_json(msg='Failed to get token: {0}'.format(e))
            token_response = None
            try:
                token_response = response.read()
                response_json = loads(token_response)
                self.oauth_token_id = response_json['id']
                self.oauth_token = response_json['token']
            except Exception as e:
                self.fail_json(msg='Failed to extract token information from login response: {0}'.format(e), **{'response': token_response})
        self.authenticated = True

    def delete_if_needed(self, existing_item, on_delete=None, auto_exit=True):
        if existing_item:
            try:
                item_url = existing_item['url']
                item_type = existing_item['type']
                item_id = existing_item['id']
                item_name = self.get_item_name(existing_item, allow_unknown=True)
            except KeyError as ke:
                self.fail_json(msg='Unable to process delete of item due to missing data {0}'.format(ke))
            response = self.delete_endpoint(item_url)
            if response['status_code'] in [202, 204]:
                if on_delete:
                    on_delete(self, response['json'])
                self.json_output['changed'] = True
                self.json_output['id'] = item_id
                self.exit_json(**self.json_output)
                if auto_exit:
                    self.exit_json(**self.json_output)
                else:
                    return self.json_output
            elif 'json' in response and '__all__' in response['json']:
                self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']['__all__'][0]))
            elif 'json' in response:
                if 'error' in response['json']:
                    self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']['error']))
                else:
                    self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['json']))
            else:
                self.fail_json(msg='Unable to delete {0} {1}: {2}'.format(item_type, item_name, response['status_code']))
        elif auto_exit:
            self.exit_json(**self.json_output)
        else:
            return self.json_output

    def modify_associations(self, association_endpoint, new_association_list):
        if new_association_list is None:
            return
        response = self.get_all_endpoint(association_endpoint)
        existing_associated_ids = [association['id'] for association in response['json']['results']]
        if association_endpoint.strip('/').split('/')[-1] in self.ordered_associations:
            if existing_associated_ids == new_association_list:
                return
            removal_list = existing_associated_ids
            addition_list = new_association_list
        else:
            if set(existing_associated_ids) == set(new_association_list):
                return
            removal_list = set(existing_associated_ids) - set(new_association_list)
            addition_list = set(new_association_list) - set(existing_associated_ids)
        for an_id in removal_list:
            response = self.post_endpoint(association_endpoint, **{'data': {'id': int(an_id), 'disassociate': True}})
            if response['status_code'] == 204:
                self.json_output['changed'] = True
            else:
                self.fail_json(msg='Failed to disassociate item {0}'.format(response['json'].get('detail', response['json'])))
        for an_id in addition_list:
            response = self.post_endpoint(association_endpoint, **{'data': {'id': int(an_id)}})
            if response['status_code'] == 204:
                self.json_output['changed'] = True
            else:
                self.fail_json(msg='Failed to associate item {0}'.format(response['json'].get('detail', response['json'])))

    def copy_item(self, existing_item, copy_from_name_or_id, new_item_name, endpoint=None, item_type='unknown', copy_lookup_data=None):
        if existing_item is not None:
            self.warn('A {0} with the name {1} already exists.'.format(item_type, new_item_name))
            self.json_output['changed'] = False
            self.json_output['copied'] = False
            return existing_item
        copy_from_lookup = self.get_one(endpoint, name_or_id=copy_from_name_or_id, **{'data': copy_lookup_data})
        if copy_from_lookup is None:
            self.fail_json(msg='A {0} with the name {1} was not able to be found.'.format(item_type, copy_from_name_or_id))
        if item_type == 'workflow_job_template':
            copy_get_check = self.get_endpoint(copy_from_lookup['related']['copy'])
            if copy_get_check['status_code'] in [200]:
                if copy_get_check['json']['can_copy'] and copy_get_check['json']['can_copy_without_user_input'] and (not copy_get_check['json']['templates_unable_to_copy']) and (not copy_get_check['json']['credentials_unable_to_copy']) and (not copy_get_check['json']['inventories_unable_to_copy']):
                    self.json_output['copy_checks'] = 'passed'
                else:
                    self.fail_json(msg='Unable to copy {0} {1} error: {2}'.format(item_type, copy_from_name_or_id, copy_get_check))
            else:
                self.fail_json(msg='Error accessing {0} {1} error: {2} '.format(item_type, copy_from_name_or_id, copy_get_check))
        response = self.post_endpoint(copy_from_lookup['related']['copy'], **{'data': {'name': new_item_name}})
        if response['status_code'] in [201]:
            self.json_output['id'] = response['json']['id']
            self.json_output['changed'] = True
            self.json_output['copied'] = True
            new_existing_item = response['json']
        elif 'json' in response and '__all__' in response['json']:
            self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['json']['__all__'][0]))
        elif 'json' in response:
            self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['json']))
        else:
            self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, new_item_name, response['status_code']))
        return new_existing_item

    def create_if_needed(self, existing_item, new_item, endpoint, on_create=None, auto_exit=True, item_type='unknown', associations=None):
        response = None
        if not endpoint:
            self.fail_json(msg='Unable to create new {0} due to missing endpoint'.format(item_type))
        item_url = None
        if existing_item:
            try:
                item_url = existing_item['url']
            except KeyError as ke:
                self.fail_json(msg='Unable to process create of item due to missing data {0}'.format(ke))
        else:
            item_name = self.get_item_name(new_item, allow_unknown=True)
            response = self.post_endpoint(endpoint, **{'data': new_item})
            if response['status_code'] in [200, 201]:
                self.json_output['name'] = 'unknown'
                for key in ('name', 'username', 'identifier', 'hostname'):
                    if key in response['json']:
                        self.json_output['name'] = response['json'][key]
                self.json_output['id'] = response['json']['id']
                self.json_output['changed'] = True
                item_url = response['json']['url']
            elif 'json' in response and '__all__' in response['json']:
                self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, item_name, response['json']['__all__'][0]))
            elif 'json' in response:
                self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, item_name, response['json']))
            else:
                self.fail_json(msg='Unable to create {0} {1}: {2}'.format(item_type, item_name, response['status_code']))
        if associations is not None:
            for association_type in associations:
                sub_endpoint = '{0}{1}/'.format(item_url, association_type)
                self.modify_associations(sub_endpoint, associations[association_type])
        if on_create is not None and self.json_output['changed']:
            on_create(self, response['json'])
        elif auto_exit:
            self.exit_json(**self.json_output)
        elif response is not None:
            last_data = response['json']
            return last_data
        else:
            return

    def _encrypted_changed_warning(self, field, old, warning=False):
        if not warning:
            return
        self.warn('The field {0} of {1} {2} has encrypted data and may inaccurately report task is changed.'.format(field, old.get('type', 'unknown'), old.get('id', 'unknown')))

    @staticmethod
    def has_encrypted_values(obj):
        """Returns True if JSON-like python content in obj has $encrypted$
        anywhere in the data as a value
        """
        if isinstance(obj, dict):
            for val in obj.values():
                if ControllerAPIModule.has_encrypted_values(val):
                    return True
        elif isinstance(obj, list):
            for val in obj:
                if ControllerAPIModule.has_encrypted_values(val):
                    return True
        elif obj == ControllerAPIModule.ENCRYPTED_STRING:
            return True
        return False

    @staticmethod
    def fields_could_be_same(old_field, new_field):
        """Treating $encrypted$ as a wild card,
        return False if the two values are KNOWN to be different
        return True if the two values are the same, or could potentially be the same,
        depending on the unknown $encrypted$ value or sub-values
        """
        if isinstance(old_field, dict) and isinstance(new_field, dict):
            if set(old_field.keys()) != set(new_field.keys()):
                return False
            for key in new_field.keys():
                if not ControllerAPIModule.fields_could_be_same(old_field[key], new_field[key]):
                    return False
            return True
        else:
            if old_field == ControllerAPIModule.ENCRYPTED_STRING:
                return True
            return bool(new_field == old_field)

    def objects_could_be_different(self, old, new, field_set=None, warning=False):
        if field_set is None:
            field_set = set((fd for fd in new.keys() if fd not in ('modified', 'related', 'summary_fields')))
        for field in field_set:
            new_field = new.get(field, None)
            old_field = old.get(field, None)
            if old_field != new_field:
                if self.update_secrets or not self.fields_could_be_same(old_field, new_field):
                    return True
            elif self.has_encrypted_values(new_field) or field not in new:
                if self.update_secrets or not self.fields_could_be_same(old_field, new_field):
                    self._encrypted_changed_warning(field, old, warning=warning)
                    return True
        return False

    def update_if_needed(self, existing_item, new_item, on_update=None, auto_exit=True, associations=None):
        response = None
        if existing_item:
            try:
                item_url = existing_item['url']
                item_type = existing_item['type']
                if item_type == 'user':
                    item_name = existing_item['username']
                elif item_type == 'workflow_job_template_node':
                    item_name = existing_item['identifier']
                elif item_type == 'credential_input_source':
                    item_name = existing_item['id']
                elif item_type == 'instance':
                    item_name = existing_item['hostname']
                else:
                    item_name = existing_item['name']
                item_id = existing_item['id']
            except KeyError as ke:
                self.fail_json(msg='Unable to process update of item due to missing data {0}'.format(ke))
            needs_patch = self.objects_could_be_different(existing_item, new_item)
            self.json_output['id'] = item_id
            if needs_patch:
                response = self.patch_endpoint(item_url, **{'data': new_item})
                if response['status_code'] == 200:
                    self.json_output['changed'] |= self.objects_could_be_different(existing_item, response['json'], field_set=new_item.keys(), warning=True)
                elif 'json' in response and '__all__' in response['json']:
                    self.fail_json(msg=response['json']['__all__'])
                else:
                    self.fail_json(**{'msg': 'Unable to update {0} {1}, see response'.format(item_type, item_name), 'response': response})
        else:
            raise RuntimeError('update_if_needed called incorrectly without existing_item')
        if associations is not None:
            for association_type, id_list in associations.items():
                endpoint = '{0}{1}/'.format(item_url, association_type)
                self.modify_associations(endpoint, id_list)
        if on_update is not None and self.json_output['changed']:
            if response is None:
                last_data = existing_item
            else:
                last_data = response['json']
            on_update(self, last_data)
        elif auto_exit:
            self.exit_json(**self.json_output)
        else:
            if response is None:
                last_data = existing_item
            else:
                last_data = response['json']
            return last_data

    def create_or_update_if_needed(self, existing_item, new_item, endpoint=None, item_type='unknown', on_create=None, on_update=None, auto_exit=True, associations=None):
        for key in list(new_item.keys()):
            if key in self.argument_spec:
                param_spec = self.argument_spec[key]
                if 'type' in param_spec and param_spec['type'] == 'bool':
                    if new_item[key] is None:
                        new_item.pop(key)
        if existing_item:
            return self.update_if_needed(existing_item, new_item, on_update=on_update, auto_exit=auto_exit, associations=associations)
        else:
            return self.create_if_needed(existing_item, new_item, endpoint, on_create=on_create, item_type=item_type, auto_exit=auto_exit, associations=associations)

    def logout(self):
        if self.authenticated and self.oauth_token_id:
            endpoint = self.url_prefix.rstrip('/') + '/api/v2/tokens/{0}/'.format(self.oauth_token_id)
            api_token_url = self.url._replace(path=endpoint, query=None).geturl()
            try:
                self.session.open('DELETE', api_token_url, validate_certs=self.verify_ssl, timeout=self.request_timeout, follow_redirects=True, force_basic_auth=True, url_username=self.username, url_password=self.password)
                self.oauth_token_id = None
                self.authenticated = False
            except HTTPError as he:
                try:
                    resp = he.read()
                except Exception as e:
                    resp = 'unknown {0}'.format(e)
                self.warn('Failed to release token: {0}, response: {1}'.format(he, resp))
            except Exception as e:
                self.warn('Failed to release token {0}: {1}'.format(self.oauth_token_id, e))

    def is_job_done(self, job_status):
        if job_status in ['new', 'pending', 'waiting', 'running']:
            return False
        else:
            return True

    def wait_on_url(self, url, object_name, object_type, timeout=30, interval=2):
        start = time.time()
        result = self.get_endpoint(url)
        while not result['json']['finished']:
            if timeout and timeout < time.time() - start:
                if object_type == 'legacy_job_wait':
                    self.json_output['msg'] = 'Monitoring of Job - {0} aborted due to timeout'.format(object_name)
                else:
                    self.json_output['msg'] = 'Monitoring of {0} - {1} aborted due to timeout'.format(object_type, object_name)
                self.wait_output(result)
                self.fail_json(**self.json_output)
            time.sleep(interval)
            result = self.get_endpoint(url)
            self.json_output['status'] = result['json']['status']
        if result['json']['failed']:
            if object_type == 'legacy_job_wait':
                self.json_output['msg'] = 'Job with id {0} failed'.format(object_name)
            else:
                self.json_output['msg'] = 'The {0} - {1}, failed'.format(object_type, object_name)
                self.json_output['job_data'] = result['json']
            self.wait_output(result)
            self.fail_json(**self.json_output)
        self.wait_output(result)
        return result

    def wait_output(self, response):
        for k in ('id', 'status', 'elapsed', 'started', 'finished'):
            self.json_output[k] = response['json'].get(k)

    def wait_on_workflow_node_url(self, url, object_name, object_type, timeout=30, interval=2, **kwargs):
        start = time.time()
        result = self.get_endpoint(url, **kwargs)
        while result['json']['count'] == 0:
            if timeout and timeout < time.time() - start:
                self.json_output['msg'] = 'Monitoring of {0} - {1} aborted due to timeout, {2}'.format(object_type, object_name, url)
                self.wait_output(result)
                self.fail_json(**self.json_output)
            time.sleep(interval)
            result = self.get_endpoint(url, **kwargs)
        if object_type == 'Workflow Approval':
            return result['json']['results'][0]
        else:
            revised_timeout = timeout - (time.time() - start)
            result = self.wait_on_url(url=result['json']['results'][0]['related']['job'], object_name=object_name, object_type=object_type, timeout=revised_timeout, interval=interval)
        self.json_output['job_data'] = result['json']
        return result