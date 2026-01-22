from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
class ElementSWQosPolicy(object):
    """
    Element Software QOS Policy
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), from_name=dict(required=False, type='str'), qos=dict(required=False, type='dict', options=dict(minIOPS=dict(type='int'), maxIOPS=dict(type='int'), burstIOPS=dict(type='int'))), debug=dict(required=False, type='bool', default=False)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.qos_policy_id = None
        self.debug = dict()
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)
        self.elementsw_helper = NaElementSWModule(self.sfe)
        self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_qos_policy')

    def get_qos_policy(self, name):
        """
        Get QOS Policy
        """
        policy, error = self.elementsw_helper.get_qos_policy(name)
        if error is not None:
            self.module.fail_json(msg=error, exception=traceback.format_exc())
        self.debug['current_policy'] = policy
        return policy

    def create_qos_policy(self, name, qos):
        """
        Create the QOS Policy
        """
        try:
            self.sfe.create_qos_policy(name=name, qos=qos)
        except (solidfire.common.ApiServerError, solidfire.common.ApiConnectionError) as exc:
            self.module.fail_json(msg='Error creating qos policy: %s: %s' % (name, to_native(exc)), exception=traceback.format_exc())

    def update_qos_policy(self, qos_policy_id, modify, name=None):
        """
        Update the QOS Policy if the policy already exists
        """
        options = dict(qos_policy_id=qos_policy_id)
        if name is not None:
            options['name'] = name
        if 'qos' in modify:
            options['qos'] = modify['qos']
        try:
            self.sfe.modify_qos_policy(**options)
        except (solidfire.common.ApiServerError, solidfire.common.ApiConnectionError) as exc:
            self.module.fail_json(msg='Error updating qos policy: %s: %s' % (self.parameters['from_name'] if name is not None else self.parameters['name'], to_native(exc)), exception=traceback.format_exc())

    def delete_qos_policy(self, qos_policy_id):
        """
        Delete the QOS Policy
        """
        try:
            self.sfe.delete_qos_policy(qos_policy_id=qos_policy_id)
        except (solidfire.common.ApiServerError, solidfire.common.ApiConnectionError) as exc:
            self.module.fail_json(msg='Error deleting qos policy: %s: %s' % (self.parameters['name'], to_native(exc)), exception=traceback.format_exc())

    def apply(self):
        """
        Process the create/delete/rename/modify actions for qos policy on the Element Software Cluster
        """
        modify = dict()
        current = self.get_qos_policy(self.parameters['name'])
        qos_policy_id = None if current is None else current['qos_policy_id']
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if cd_action == 'create' and self.parameters.get('from_name') is not None:
            from_qos_policy = self.get_qos_policy(self.parameters['from_name'])
            if from_qos_policy is None:
                self.module.fail_json(msg='Error renaming qos policy, no existing policy with name/id: %s' % self.parameters['from_name'])
            cd_action = 'rename'
            qos_policy_id = from_qos_policy['qos_policy_id']
            self.na_helper.changed = True
            modify = self.na_helper.get_modified_attributes(from_qos_policy, self.parameters)
        if cd_action == 'create' and 'qos' not in self.parameters:
            self.module.fail_json(msg="Error creating qos policy: %s, 'qos:' option is required" % self.parameters['name'])
        self.debug['modify'] = modify
        if not self.module.check_mode:
            if cd_action == 'create':
                self.create_qos_policy(self.parameters['name'], self.parameters['qos'])
            elif cd_action == 'delete':
                self.delete_qos_policy(qos_policy_id)
            elif cd_action == 'rename':
                self.update_qos_policy(qos_policy_id, modify, name=self.parameters['name'])
            elif modify:
                self.update_qos_policy(qos_policy_id, modify)
        results = dict(changed=self.na_helper.changed)
        if self.parameters['debug']:
            results['debug'] = self.debug
        self.module.exit_json(**results)