from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapSecurityConfig:
    """
        Modifies SSL Security Config
    """

    def __init__(self):
        """
            Initialize the ONTAP Security Config class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=False, type='str', default='ssl'), is_fips_enabled=dict(required=False, type='bool'), supported_ciphers=dict(required=False, type='str'), supported_protocols=dict(required=False, type='list', elements='str', choices=['TLSv1.3', 'TLSv1.2', 'TLSv1.1', 'TLSv1']), supported_cipher_suites=dict(required=False, type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        unsupported_rest_properties = ['supported_ciphers']
        partially_supported_rest_properties = [['supported_cipher_suites', (9, 10, 1)], ['supported_protocols', (9, 10, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if self.use_rest and self.rest_api.fail_if_not_rest_minimum_version('na_ontap_security_config', 9, 8, 0):
            msg = 'REST requires ONTAP 9.8 or later.'
            self.use_rest = self.na_helper.fall_back_to_zapi(self.module, msg, self.parameters)
        if not self.use_rest:
            if self.parameters.get('supported_cipher_suites'):
                self.module.fail_json(msg='Error: The option supported_cipher_suites is supported only with REST.')
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg='The python NetApp-Lib module is required')
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)
            if 'is_fips_enabled' in self.parameters and 'supported_ciphers' in self.parameters:
                if self.parameters['is_fips_enabled']:
                    self.module.fail_json(msg='is_fips_enabled was specified as true and supported_ciphers was specified.                         If fips is enabled then supported ciphers should not be specified')
            if 'supported_ciphers' in self.parameters:
                self.parameters['supported_ciphers'] = self.parameters['supported_ciphers'].replace('\\', '')
        if 'is_fips_enabled' in self.parameters and 'supported_protocols' in self.parameters:
            if self.parameters['is_fips_enabled'] and 'TLSv1' in self.parameters['supported_protocols']:
                self.module.fail_json(msg='is_fips_enabled was specified as true and TLSv1 was specified as a supported protocol.                     If fips is enabled then TLSv1 is not a supported protocol')
            if self.parameters['is_fips_enabled'] and 'TLSv1.1' in self.parameters['supported_protocols']:
                self.module.fail_json(msg='is_fips_enabled was specified as true and TLSv1.1 was specified as a supported protocol.                     If fips is enabled then TLSv1.1 is not a supported protocol')

    def get_security_config(self):
        """
            Get the current security configuration
        """
        if self.use_rest:
            return self.get_security_config_rest()
        return_value = None
        security_config_get_iter = netapp_utils.zapi.NaElement('security-config-get')
        security_config_info = netapp_utils.zapi.NaElement('desired-attributes')
        if 'is_fips_enabled' in self.parameters:
            security_config_info.add_new_child('is-fips-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_fips_enabled']))
        if 'supported_ciphers' in self.parameters:
            security_config_info.add_new_child('supported-ciphers', self.parameters['supported_ciphers'])
        if 'supported_protocols' in self.parameters:
            security_config_info.add_new_child('supported-protocols', ','.join(self.parameters['supported_protocols']))
        security_config_get_iter.add_child_elem(security_config_info)
        security_config_get_iter.add_new_child('interface', self.parameters['name'])
        try:
            result = self.server.invoke_successfully(security_config_get_iter, True)
            security_supported_protocols = []
            if result.get_child_by_name('attributes'):
                attributes = result.get_child_by_name('attributes')
                security_config_attributes = attributes.get_child_by_name('security-config-info')
                supported_protocols = security_config_attributes.get_child_by_name('supported-protocols')
                for supported_protocol in supported_protocols.get_children():
                    security_supported_protocols.append(supported_protocol.get_content())
                return_value = {'name': security_config_attributes['interface'], 'is_fips_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=security_config_attributes['is-fips-enabled']), 'supported_ciphers': security_config_attributes['supported-ciphers'], 'supported_protocols': security_supported_protocols}
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting security config for interface %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return return_value

    def modify_security_config(self, modify):
        """
        Modifies the security configuration.
        """
        if self.use_rest:
            return self.modify_security_config_rest(modify)
        security_config_obj = netapp_utils.zapi.NaElement('security-config-modify')
        security_config_obj.add_new_child('interface', self.parameters['name'])
        if 'is_fips_enabled' in self.parameters:
            self.parameters['is_fips_enabled'] = self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_fips_enabled'])
            security_config_obj.add_new_child('is-fips-enabled', self.parameters['is_fips_enabled'])
        if 'supported_ciphers' in self.parameters:
            security_config_obj.add_new_child('supported-ciphers', self.parameters['supported_ciphers'])
        if 'supported_protocols' in self.parameters:
            supported_protocol_obj = netapp_utils.zapi.NaElement('supported-protocols')
            for protocol in self.parameters['supported_protocols']:
                supported_protocol_obj.add_new_child('string', protocol)
            security_config_obj.add_child_elem(supported_protocol_obj)
        try:
            self.server.invoke_successfully(security_config_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying security config for interface %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def get_security_config_rest(self):
        """
            Get the current security configuration
        """
        fields = 'fips.enabled,'
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            fields += 'tls.cipher_suites,tls.protocol_versions'
        record, error = rest_generic.get_one_record(self.rest_api, '/security', None, fields)
        if error:
            self.module.fail_json(msg='Error on getting security config: %s' % error)
        if record:
            return {'is_fips_enabled': self.na_helper.safe_get(record, ['fips', 'enabled']), 'supported_cipher_suites': self.na_helper.safe_get(record, ['tls', 'cipher_suites']), 'supported_protocols': self.na_helper.safe_get(record, ['tls', 'protocol_versions'])}
        return record

    def modify_security_config_rest(self, modify):
        """
            Modify the current security configuration
        """
        body = {}
        if 'is_fips_enabled' in modify:
            body['fips.enabled'] = modify['is_fips_enabled']
        if 'supported_cipher_suites' in modify:
            body['tls.cipher_suites'] = modify['supported_cipher_suites']
        if 'supported_protocols' in modify:
            body['tls.protocol_versions'] = modify['supported_protocols']
        record, error = rest_generic.patch_async(self.rest_api, '/security', None, body)
        if error:
            self.module.fail_json(msg='Error on modifying security config: %s' % error)

    def apply(self):
        current = self.get_security_config()
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_security_config(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, modify=modify)
        self.module.exit_json(**result)