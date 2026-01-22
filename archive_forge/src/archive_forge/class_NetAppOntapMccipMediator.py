from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
class NetAppOntapMccipMediator(object):
    """
    Mediator object for Add/Remove/Display
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), mediator_address=dict(required=True, type='str'), mediator_user=dict(required=True, type='str'), mediator_password=dict(required=True, type='str', no_log=True)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            self.module.fail_json(msg=self.rest_api.requires_ontap_9_6('na_ontap_mcc_mediator'))

    def add_mediator(self):
        """
        Adds an ONTAP Mediator to MCC configuration
        """
        api = 'cluster/mediators'
        params = {'ip_address': self.parameters['mediator_address'], 'password': self.parameters['mediator_password'], 'user': self.parameters['mediator_user']}
        dummy, error = self.rest_api.post(api, params)
        if error:
            self.module.fail_json(msg=error)

    def remove_mediator(self, current_uuid):
        """
        Removes the ONTAP Mediator from MCC configuration
        """
        api = 'cluster/mediators/%s' % current_uuid
        params = {'ip_address': self.parameters['mediator_address'], 'password': self.parameters['mediator_password'], 'user': self.parameters['mediator_user']}
        dummy, error = self.rest_api.delete(api, params)
        if error:
            self.module.fail_json(msg=error)

    def get_mediator(self):
        """
        Determine if the MCC configuration has added an ONTAP Mediator
        """
        api = 'cluster/mediators'
        message, error = self.rest_api.get(api, None)
        if error:
            self.module.fail_json(msg=error)
        if message['num_records'] > 0:
            return message['records'][0]['uuid']
        return None

    def apply(self):
        """
        Apply action to MCC Mediator
        """
        current = self.get_mediator()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'create':
                self.add_mediator()
            elif cd_action == 'delete':
                self.remove_mediator(current)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)