from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridNtp(object):
    """
    Create, modify and delete NTP entries for StorageGRID
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), ntp_servers=dict(required=True, type='list', elements='str'), passphrase=dict(required=True, type='str', no_log=True)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = self.parameters['ntp_servers']
        self.passphrase = self.parameters['passphrase']
        self.ntp_input = {'passphrase': self.passphrase, 'servers': self.data}

    def get_grid_ntp(self):
        api = 'api/v3/grid/ntp-servers'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def update_grid_ntp(self):
        api = 'api/v3/grid/ntp-servers/update'
        response, error = self.rest_api.post(api, self.ntp_input)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        grid_ntp = self.get_grid_ntp()
        cd_action = self.na_helper.get_cd_action(grid_ntp, self.parameters['ntp_servers'])
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            ntp_diff = [i for i in self.data + grid_ntp if i not in self.data or i not in grid_ntp]
            if ntp_diff:
                update = True
            if update:
                self.na_helper.changed = True
        result_message = ''
        resp_data = grid_ntp
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            else:
                resp_data = self.update_grid_ntp()
                result_message = 'Grid NTP updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)