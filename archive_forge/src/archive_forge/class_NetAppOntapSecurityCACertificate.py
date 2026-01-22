from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapSecurityCACertificate:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), svm=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_security_ipsec_ca_certificate', 9, 10, 1)

    def get_certificate_uuid(self):
        """Get certificate UUID."""
        api = 'security/certificates'
        query = {'name': self.parameters['name']}
        if self.parameters.get('svm'):
            query['svm.name'] = self.parameters['svm']
        else:
            query['scope'] = 'cluster'
        record, error = rest_generic.get_one_record(self.rest_api, api, query, 'uuid')
        if error:
            self.module.fail_json(msg='Error fetching uuid for certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if record:
            return record['uuid']
        return None

    def get_ipsec_ca_certificate(self):
        """GET IPsec CA certificate record"""
        self.uuid = self.get_certificate_uuid()
        if self.uuid is None:
            if self.parameters['state'] == 'absent':
                return None
            svm_or_scope = self.parameters['svm'] if self.parameters.get('svm') else 'cluster'
            self.module.fail_json(msg='Error: certificate %s is not installed in %s' % (self.parameters['name'], svm_or_scope))
        api = 'security/ipsec/ca-certificates/%s' % self.uuid
        record, error = rest_generic.get_one_record(self.rest_api, api)
        if error:
            if "entry doesn't exist" in error:
                return None
            self.module.fail_json(msg='Error fetching security IPsec CA certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return record if record else None

    def create_ipsec_ca_certificate(self):
        """Create IPsec CA certifcate"""
        api = 'security/ipsec/ca-certificates'
        body = {'certificate.uuid': self.uuid}
        if self.parameters.get('svm'):
            body['svm.name'] = self.parameters['svm']
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error adding security IPsec CA certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_ipsec_ca_certificate(self):
        """Delete IPSec CA certificate"""
        api = 'security/ipsec/ca-certificates'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid)
        if error:
            self.module.fail_json(msg='Error deleting security IPsec CA certificate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        current = self.get_ipsec_ca_certificate()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_ipsec_ca_certificate()
            else:
                self.delete_ipsec_ca_certificate()
        self.module.exit_json(changed=self.na_helper.changed)