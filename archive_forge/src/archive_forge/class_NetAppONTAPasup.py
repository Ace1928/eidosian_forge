from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPasup:
    """Class with autosupport methods"""

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), node_name=dict(required=True, type='str'), transport=dict(required=False, type='str', choices=['smtp', 'http', 'https']), noteto=dict(required=False, type='list', elements='str'), post_url=dict(required=False, type='str'), support=dict(required=False, type='bool'), mail_hosts=dict(required=False, type='list', elements='str'), from_address=dict(required=False, type='str'), partner_addresses=dict(required=False, type='list', elements='str'), to_addresses=dict(required=False, type='list', elements='str'), proxy_url=dict(required=False, type='str', no_log=True), hostname_in_subject=dict(required=False, type='bool'), nht_data_enabled=dict(required=False, type='bool'), perf_data_enabled=dict(required=False, type='bool'), retry_count=dict(required=False, type='int'), reminder_enabled=dict(required=False, type='bool'), max_http_size=dict(required=False, type='int'), max_smtp_size=dict(required=False, type='int'), private_data_removed=dict(required=False, type='bool'), local_collection_enabled=dict(required=False, type='bool'), ondemand_enabled=dict(required=False, type='bool'), validate_digital_certificate=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.parameters['service_state'] = 'started' if self.parameters['state'] == 'present' else 'stopped'
        self.set_playbook_zapi_key_map()
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def set_playbook_zapi_key_map(self):
        self.na_helper.zapi_string_keys = {'node_name': 'node-name', 'transport': 'transport', 'post_url': 'post-url', 'from_address': 'from', 'proxy_url': 'proxy-url'}
        self.na_helper.zapi_int_keys = {'retry_count': 'retry-count', 'max_http_size': 'max-http-size', 'max_smtp_size': 'max-smtp-size'}
        self.na_helper.zapi_list_keys = {'noteto': ('noteto', 'mail-address'), 'mail_hosts': ('mail-hosts', 'string'), 'partner_addresses': ('partner-address', 'mail-address'), 'to_addresses': ('to', 'mail-address')}
        self.na_helper.zapi_bool_keys = {'support': 'is-support-enabled', 'hostname_in_subject': 'is-node-in-subject', 'nht_data_enabled': 'is-nht-data-enabled', 'perf_data_enabled': 'is-perf-data-enabled', 'reminder_enabled': 'is-reminder-enabled', 'private_data_removed': 'is-private-data-removed', 'local_collection_enabled': 'is-local-collection-enabled', 'ondemand_enabled': 'is-ondemand-enabled', 'validate_digital_certificate': 'validate-digital-certificate'}

    def get_autosupport_config(self):
        """
        get current autosupport details
        :return: dict()
        """
        asup_info = {}
        if self.use_rest:
            api = 'private/cli/system/node/autosupport'
            query = {'node': self.parameters['node_name'], 'fields': 'state,node,transport,noteto,url,support,mail-hosts,from,partner-address,to,proxy-url,hostname-subj,nht,perf,retry-count,reminder,max-http-size,max-smtp-size,remove-private-data,ondemand-server-url,support,reminder,ondemand-state,local-collection,validate-digital-certificate'}
            record, error = rest_generic.get_one_record(self.rest_api, api, query)
            if error:
                self.module.fail_json(msg='Error fetching info: %s' % error)
            for param in ('transport', 'mail_hosts', 'proxy_url', 'retry_count', 'max_http_size', 'max_smtp_size', 'noteto', 'validate_digital_certificate'):
                if param in record:
                    asup_info[param] = record[param]
            asup_info['support'] = record['support'] in ['enable', True]
            asup_info['node_name'] = record['node'] if 'node' in record else ''
            asup_info['post_url'] = record['url'] if 'url' in record else ''
            asup_info['from_address'] = record['from'] if 'from' in record else ''
            asup_info['to_addresses'] = record['to'] if 'to' in record else list()
            asup_info['hostname_in_subject'] = record['hostname_subj'] if 'hostname_subj' in record else False
            asup_info['nht_data_enabled'] = record['nht'] if 'nht' in record else False
            asup_info['perf_data_enabled'] = record['perf'] if 'perf' in record else False
            asup_info['reminder_enabled'] = record['reminder'] if 'reminder' in record else False
            asup_info['private_data_removed'] = record['remove_private_data'] if 'remove_private_data' in record else False
            asup_info['local_collection_enabled'] = record['local_collection'] if 'local_collection' in record else False
            asup_info['ondemand_enabled'] = record['ondemand_state'] in ['enable', True] if 'ondemand_state' in record else False
            asup_info['service_state'] = 'started' if record['state'] in ['enable', True] else 'stopped'
            asup_info['partner_addresses'] = record['partner_address'] if 'partner_address' in record else list()
        else:
            asup_details = netapp_utils.zapi.NaElement('autosupport-config-get')
            asup_details.add_new_child('node-name', self.parameters['node_name'])
            try:
                result = self.server.invoke_successfully(asup_details, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error fetching info: %s' % to_native(error), exception=traceback.format_exc())
            asup_attr_info = result.get_child_by_name('attributes').get_child_by_name('autosupport-config-info')
            asup_info['service_state'] = 'started' if asup_attr_info['is-enabled'] == 'true' else 'stopped'
            for item_key, zapi_key in self.na_helper.zapi_string_keys.items():
                value = asup_attr_info.get_child_content(zapi_key)
                asup_info[item_key] = value if value is not None else ''
            for item_key, zapi_key in self.na_helper.zapi_int_keys.items():
                value = asup_attr_info.get_child_content(zapi_key)
                if value is not None:
                    asup_info[item_key] = self.na_helper.get_value_for_int(from_zapi=True, value=value)
            for item_key, zapi_key in self.na_helper.zapi_bool_keys.items():
                value = asup_attr_info.get_child_content(zapi_key)
                if value is not None:
                    asup_info[item_key] = self.na_helper.get_value_for_bool(from_zapi=True, value=value)
            for item_key, zapi_key in self.na_helper.zapi_list_keys.items():
                parent, dummy = zapi_key
                asup_info[item_key] = self.na_helper.get_value_for_list(from_zapi=True, zapi_parent=asup_attr_info.get_child_by_name(parent))
        return asup_info

    def modify_autosupport_config(self, modify):
        """
        modify autosupport config
        @return: modfied attributes / FAILURE with an error_message
        """
        if self.use_rest:
            api = 'private/cli/system/node/autosupport'
            query = {'node': self.parameters['node_name']}
            if 'service_state' in modify:
                modify['state'] = modify['service_state'] == 'started'
                del modify['service_state']
            if 'post_url' in modify:
                modify['url'] = modify.pop('post_url')
            if 'from_address' in modify:
                modify['from'] = modify.pop('from_address')
            if 'to_addresses' in modify:
                modify['to'] = modify.pop('to_addresses')
            if 'hostname_in_subject' in modify:
                modify['hostname_subj'] = modify.pop('hostname_in_subject')
            if 'nht_data_enabled' in modify:
                modify['nht'] = modify.pop('nht_data_enabled')
            if 'perf_data_enabled' in modify:
                modify['perf'] = modify.pop('perf_data_enabled')
            if 'reminder_enabled' in modify:
                modify['reminder'] = modify.pop('reminder_enabled')
            if 'private_data_removed' in modify:
                modify['remove_private_data'] = modify.pop('private_data_removed')
            if 'local_collection_enabled' in modify:
                modify['local_collection'] = modify.pop('local_collection_enabled')
            if 'ondemand_enabled' in modify:
                modify['ondemand_state'] = modify.pop('ondemand_enabled')
            if 'partner_addresses' in modify:
                modify['partner_address'] = modify.pop('partner_addresses')
            dummy, error = rest_generic.patch_async(self.rest_api, api, None, modify, query)
            if error:
                self.module.fail_json(msg='Error modifying asup: %s' % error)
        else:
            asup_details = {'node-name': self.parameters['node_name']}
            if modify.get('service_state'):
                asup_details['is-enabled'] = 'true' if modify.get('service_state') == 'started' else 'false'
            asup_config = netapp_utils.zapi.NaElement('autosupport-config-modify')
            for item_key in modify:
                if item_key in self.na_helper.zapi_string_keys:
                    zapi_key = self.na_helper.zapi_string_keys.get(item_key)
                    asup_details[zapi_key] = modify[item_key]
                elif item_key in self.na_helper.zapi_int_keys:
                    zapi_key = self.na_helper.zapi_int_keys.get(item_key)
                    asup_details[zapi_key] = modify[item_key]
                elif item_key in self.na_helper.zapi_bool_keys:
                    zapi_key = self.na_helper.zapi_bool_keys.get(item_key)
                    asup_details[zapi_key] = self.na_helper.get_value_for_bool(from_zapi=False, value=modify[item_key])
                elif item_key in self.na_helper.zapi_list_keys:
                    parent_key, child_key = self.na_helper.zapi_list_keys.get(item_key)
                    asup_config.add_child_elem(self.na_helper.get_value_for_list(from_zapi=False, zapi_parent=parent_key, zapi_child=child_key, data=modify.get(item_key)))
            asup_config.translate_struct(asup_details)
            try:
                return self.server.invoke_successfully(asup_config, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error modifying asup: %s' % to_native(error), exception=traceback.format_exc())

    @staticmethod
    def strip_password(url):
        """ if url matches user:password@address return user@address
            otherwise return None
        """
        if url:
            needle = '(.*):(.*)@(.*)'
            matched = re.match(needle, url)
            if matched:
                return matched.group(1, 3)
        return (None, None)

    def idempotency_check(self, current, modify):
        sanitized_modify = dict(modify)
        if 'proxy_url' in modify:
            user_url_m = self.strip_password(modify['proxy_url'])
            user_url_c = self.strip_password(current.get('proxy_url'))
            if user_url_m == user_url_c and user_url_m != (None, None):
                self.module.warn('na_ontap_autosupport is not idempotent because the password value in proxy_url cannot be compared.')
            if user_url_m != (None, None):
                sanitized_modify['proxy_url'] = '%s:XXXXXXXX@%s' % user_url_m
        return sanitized_modify

    def apply(self):
        """
        Apply action to autosupport
        """
        current = self.get_autosupport_config()
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        sanitized_modify = self.idempotency_check(current, modify)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_autosupport_config(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, modify=sanitized_modify)
        self.module.exit_json(**result)