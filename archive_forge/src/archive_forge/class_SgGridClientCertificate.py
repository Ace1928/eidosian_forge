from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridClientCertificate:
    """
    Update StorageGRID client certificates
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), certificate_id=dict(required=False, type='str'), display_name=dict(required=False, type='str'), public_key=dict(required=False, type='str'), allow_prometheus=dict(required=False, type='bool')))
        parameter_map = {'display_name': 'displayName', 'public_key': 'publicKey', 'allow_prometheus': 'allowPrometheus'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['display_name', 'public_key'])], required_one_of=[('display_name', 'certificate_id')], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        if self.parameters['state'] == 'present':
            for k in parameter_map.keys():
                if self.parameters.get(k) is not None:
                    self.data[parameter_map[k]] = self.parameters[k]
        self.module.fail_json

    def get_grid_client_certificate_id(self):
        api = 'api/v3/grid/client-certificates'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        for cert in response.get('data'):
            if cert['displayName'] == self.parameters['display_name']:
                return cert['id']
        return None

    def get_grid_client_certificate(self, cert_id):
        api = 'api/v3/grid/client-certificates/%s' % cert_id
        account, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        else:
            return account['data']
        return None

    def create_grid_client_certificate(self):
        api = 'api/v3/grid/client-certificates'
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error['text'])
        return response['data']

    def delete_grid_client_certificate(self, cert_id):
        api = 'api/v3/grid/client-certificates/' + cert_id
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def update_grid_client_certificate(self, cert_id):
        api = 'api/v3/grid/client-certificates/' + cert_id
        response, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error['text'])
        return response['data']

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        client_certificate = None
        if self.parameters.get('certificate_id'):
            client_certificate = self.get_grid_client_certificate(self.parameters['certificate_id'])
        else:
            client_cert_id = self.get_grid_client_certificate_id()
            if client_cert_id:
                client_certificate = self.get_grid_client_certificate(client_cert_id)
        cd_action = self.na_helper.get_cd_action(client_certificate, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(client_certificate, self.data)
        result_message = ''
        resp_data = client_certificate
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'delete':
                self.delete_grid_client_certificate(client_certificate['id'])
                result_message = 'Client Certificate deleted'
            elif cd_action == 'create':
                resp_data = self.create_grid_client_certificate()
                result_message = 'Client Certificate created'
            elif modify:
                resp_data = self.update_grid_client_certificate(client_certificate['id'])
                result_message = 'Client Certificate updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)