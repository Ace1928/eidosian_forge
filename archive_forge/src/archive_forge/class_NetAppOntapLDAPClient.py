from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
class NetAppOntapLDAPClient:
    """
    LDAP Client definition class
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(ad_domain=dict(required=False, default=None, type='str'), base_dn=dict(required=False, type='str'), base_scope=dict(required=False, default=None, choices=['subtree', 'onelevel', 'base']), bind_as_cifs_server=dict(required=False, type='bool'), bind_dn=dict(required=False, default=None, type='str'), bind_password=dict(type='str', required=False, default=None, no_log=True), name=dict(required=False, type='str'), servers=dict(required=False, type='list', elements='str', aliases=['ldap_servers']), min_bind_level=dict(required=False, default=None, choices=['anonymous', 'simple', 'sasl']), preferred_ad_servers=dict(required=False, type='list', elements='str'), port=dict(required=False, type='int', aliases=['tcp_port']), query_timeout=dict(required=False, default=None, type='int'), referral_enabled=dict(required=False, type='bool'), schema=dict(required=False, type='str'), session_security=dict(required=False, default=None, choices=['none', 'sign', 'seal']), state=dict(required=False, choices=['present', 'absent'], default='present'), use_start_tls=dict(required=False, type='bool'), vserver=dict(required=True, type='str'), ldaps_enabled=dict(required=False, type='bool'), skip_config_validation=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, required_if=[('state', 'present', ['schema'])], mutually_exclusive=[['servers', 'ad_domain'], ['servers', 'preferred_ad_servers'], ['use_start_tls', 'ldaps_enabled']])
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        unsupported_rest_properties = ['name']
        partially_supported_rest_properties = [['bind_as_cifs_server', (9, 9, 0)], ['query_timeout', (9, 9, 0)], ['referral_enabled', (9, 9, 0)], ['ldaps_enabled', (9, 9, 0)], ['skip_config_validation', (9, 9, 0)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
            if not self.parameters.get('name'):
                self.module.fail_json(msg='Error: name is a required field with ZAPI.')
        self.simple_attributes = ['ad_domain', 'base_dn', 'base_scope', 'bind_as_cifs_server', 'bind_dn', 'bind_password', 'min_bind_level', 'tcp_port', 'query_timeout', 'referral_enabled', 'session_security', 'use_start_tls', 'ldaps_enabled']

    def get_ldap_client(self, client_config_name=None, vserver_name=None):
        """
        Checks if LDAP client config exists.

        :return:
            ldap client config object if found
            None if not found
        :rtype: object/None
        """
        client_config_info = netapp_utils.zapi.NaElement('ldap-client-get-iter')
        if client_config_name is None:
            client_config_name = self.parameters['name']
        if vserver_name is None:
            vserver_name = '*'
        query_details = netapp_utils.zapi.NaElement.create_node_with_children('ldap-client', **{'ldap-client-config': client_config_name, 'vserver': vserver_name})
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(query_details)
        client_config_info.add_child_elem(query)
        result = self.server.invoke_successfully(client_config_info, enable_tunneling=False)
        client_config_details = None
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            attributes_list = result.get_child_by_name('attributes-list')
            client_config_info = attributes_list.get_child_by_name('ldap-client')
            ldap_server_list = self.get_list_from_children(client_config_info, 'ldap-servers')
            preferred_ad_servers_list = self.get_list_from_children(client_config_info, 'preferred-ad-servers')
            client_config_details = {'name': client_config_info.get_child_content('ldap-client-config'), 'servers': ldap_server_list, 'ad_domain': client_config_info.get_child_content('ad-domain'), 'base_dn': client_config_info.get_child_content('base-dn'), 'base_scope': client_config_info.get_child_content('base-scope'), 'bind_as_cifs_server': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('bind-as-cifs-server')), 'bind_dn': client_config_info.get_child_content('bind-dn'), 'bind_password': client_config_info.get_child_content('bind-password'), 'min_bind_level': client_config_info.get_child_content('min-bind-level'), 'tcp_port': self.na_helper.get_value_for_int(from_zapi=True, value=client_config_info.get_child_content('tcp-port')), 'preferred_ad_servers': preferred_ad_servers_list, 'query_timeout': self.na_helper.get_value_for_int(from_zapi=True, value=client_config_info.get_child_content('query-timeout')), 'referral_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('referral-enabled')), 'schema': client_config_info.get_child_content('schema'), 'session_security': client_config_info.get_child_content('session-security'), 'use_start_tls': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('use-start-tls')), 'ldaps_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=client_config_info.get_child_content('ldaps-enabled'))}
        return client_config_details

    def get_list_from_children(self, client_config_info, element_name):
        get_list = client_config_info.get_child_by_name(element_name)
        return [x.get_content() for x in get_list.get_children()] if get_list is not None else []

    def create_ldap_client(self):
        """
        Create LDAP client configuration
        """
        options = {'ldap-client-config': self.parameters['name'], 'schema': self.parameters['schema']}
        for attribute in self.simple_attributes:
            if self.parameters.get(attribute) is not None:
                options[str(attribute).replace('_', '-')] = str(self.parameters[attribute])
        ldap_client_create = netapp_utils.zapi.NaElement.create_node_with_children('ldap-client-create', **options)
        if self.parameters.get('servers') is not None:
            self.add_element_with_children('ldap-servers', 'servers', 'string', ldap_client_create)
        if self.parameters.get('preferred_ad_servers') is not None:
            self.add_element_with_children('preferred-ad-servers', 'preferred_ad_servers', 'ip-address', ldap_client_create)
        try:
            self.server.invoke_successfully(ldap_client_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error creating LDAP client %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())

    def add_element_with_children(self, element_name, param_name, child_name, ldap_client_create):
        ldap_servers_element = netapp_utils.zapi.NaElement(element_name)
        for ldap_server_name in self.parameters[param_name]:
            ldap_servers_element.add_new_child(child_name, ldap_server_name)
        ldap_client_create.add_child_elem(ldap_servers_element)

    def delete_ldap_client(self):
        """
        Delete LDAP client configuration
        """
        ldap_client_delete = netapp_utils.zapi.NaElement.create_node_with_children('ldap-client-delete', **{'ldap-client-config': self.parameters['name']})
        try:
            self.server.invoke_successfully(ldap_client_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error deleting LDAP client configuration %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())

    def modify_ldap_client(self, modify):
        """
        Modify LDAP client
        :param modify: list of modify attributes
        """
        ldap_client_modify = netapp_utils.zapi.NaElement('ldap-client-modify')
        ldap_client_modify.add_new_child('ldap-client-config', self.parameters['name'])
        for attribute in modify:
            if attribute == 'servers':
                self.add_element_with_children('ldap-servers', attribute, 'string', ldap_client_modify)
            if attribute == 'preferred_ad_servers':
                self.add_element_with_children('preferred-ad-servers', attribute, 'ip-address', ldap_client_modify)
            if attribute in self.simple_attributes:
                ldap_client_modify.add_new_child(str(attribute).replace('_', '-'), str(self.parameters[attribute]))
        try:
            self.server.invoke_successfully(ldap_client_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error modifying LDAP client %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())

    def get_ldap_client_rest(self):
        """
        Retrives ldap client config with rest API.
        """
        if not self.use_rest:
            return self.get_ldap_client()
        query = {'svm.name': self.parameters.get('vserver'), 'fields': 'svm.uuid,ad_domain,servers,preferred_ad_servers,bind_dn,schema,port,base_dn,base_scope,min_bind_level,session_security,use_start_tls,'}
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 0):
            query['fields'] += 'bind_as_cifs_server,query_timeout,referral_enabled,ldaps_enabled'
        record, error = rest_generic.get_one_record(self.rest_api, 'name-services/ldap', query)
        if error:
            self.module.fail_json(msg='Error on getting idap client info: %s' % error)
        if record:
            return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'ad_domain': self.na_helper.safe_get(record, ['ad_domain']), 'preferred_ad_servers': self.na_helper.safe_get(record, ['preferred_ad_servers']), 'servers': self.na_helper.safe_get(record, ['servers']), 'schema': self.na_helper.safe_get(record, ['schema']), 'port': self.na_helper.safe_get(record, ['port']), 'ldaps_enabled': self.na_helper.safe_get(record, ['ldaps_enabled']), 'min_bind_level': self.na_helper.safe_get(record, ['min_bind_level']), 'bind_dn': self.na_helper.safe_get(record, ['bind_dn']), 'base_dn': self.na_helper.safe_get(record, ['base_dn']), 'base_scope': self.na_helper.safe_get(record, ['base_scope']), 'use_start_tls': self.na_helper.safe_get(record, ['use_start_tls']), 'session_security': self.na_helper.safe_get(record, ['session_security']), 'referral_enabled': self.na_helper.safe_get(record, ['referral_enabled']), 'bind_as_cifs_server': self.na_helper.safe_get(record, ['bind_as_cifs_server']), 'query_timeout': self.na_helper.safe_get(record, ['query_timeout'])}
        return None

    def create_ldap_client_body_rest(self, modify=None):
        """
        ldap client config body for create and modify with rest API.
        """
        config_options = ['ad_domain', 'servers', 'preferred_ad_servers', 'bind_dn', 'schema', 'port', 'base_dn', 'referral_enabled', 'ldaps_enabled', 'base_scope', 'bind_as_cifs_server', 'bind_password', 'min_bind_level', 'query_timeout', 'session_security', 'use_start_tls']
        processing_options = ['skip_config_validation']
        body = {}
        for key in config_options:
            if not modify and key in self.parameters:
                body[key] = self.parameters[key]
            elif modify and key in modify:
                body[key] = modify[key]
        for key in processing_options:
            if body and key in self.parameters:
                body[key] = self.parameters[key]
        return body

    def create_ldap_client_rest(self):
        """
        create ldap client config with rest API.
        """
        if not self.use_rest:
            return self.create_ldap_client()
        body = self.create_ldap_client_body_rest()
        body['svm.name'] = self.parameters['vserver']
        api = 'name-services/ldap'
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error on creating ldap client: %s' % error)

    def delete_ldap_client_rest(self, current):
        """
        delete ldap client config with rest API.
        """
        if not self.use_rest:
            return self.delete_ldap_client()
        api = 'name-services/ldap'
        dummy, error = rest_generic.delete_async(self.rest_api, api, current['svm']['uuid'], body=None)
        if error is not None:
            self.module.fail_json(msg='Error on deleting ldap client rest: %s' % error)

    def modify_ldap_client_rest(self, current, modify):
        """
        modif ldap client config with rest API.
        """
        if not self.use_rest:
            return self.modify_ldap_client(modify)
        body = self.create_ldap_client_body_rest(modify)
        if body:
            api = 'name-services/ldap'
            dummy, error = rest_generic.patch_async(self.rest_api, api, current['svm']['uuid'], body)
            if error is not None:
                self.module.fail_json(msg='Error on modifying ldap client config: %s' % error)

    def apply(self):
        """Call create/modify/delete operations."""
        current = self.get_ldap_client_rest()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.parameters['state'] == 'present' and (not self.parameters.get('servers')) and (self.parameters.get('ad_domain') is None):
            self.module.fail_json(msg='Required one of servers or ad_domain')
        if cd_action == 'create' and self.use_rest:
            rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_ldap_client_rest()
            elif cd_action == 'delete':
                self.delete_ldap_client_rest(current)
            elif modify:
                self.modify_ldap_client_rest(current, modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)