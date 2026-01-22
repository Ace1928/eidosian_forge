from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
class AzureRMNetAppSnapshot(AzureRMNetAppModuleBase):
    """ crate or delete snapshots """

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), volume_name=dict(type='str', required=True), pool_name=dict(type='str', required=True), account_name=dict(type='str', required=True), location=dict(type='str', required=False), state=dict(choices=['present', 'absent'], default='present', type='str'))
        self.na_helper = NetAppModule()
        self.parameters = dict()
        super(AzureRMNetAppSnapshot, self).__init__(derived_arg_spec=self.module_arg_spec, required_if=[('state', 'present', ['location'])], supports_check_mode=True, supports_tags=False)

    def get_azure_netapp_snapshot(self):
        """
            Returns snapshot object for an existing snapshot
            Return None if snapshot does not exist
        """
        try:
            snapshot_get = self.netapp_client.snapshots.get(self.parameters['resource_group'], self.parameters['account_name'], self.parameters['pool_name'], self.parameters['volume_name'], self.parameters['name'])
        except (CloudError, ResourceNotFoundError):
            return None
        return snapshot_get

    def create_azure_netapp_snapshot(self):
        """
            Create a snapshot for the given Azure NetApp Account
            :return: None
        """
        kw_args = dict(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['volume_name'], snapshot_name=self.parameters['name'])
        if self.new_style:
            kw_args['body'] = Snapshot(location=self.parameters['location'])
        else:
            kw_args['location'] = self.parameters['location']
        try:
            result = self.get_method('snapshots', 'create')(**kw_args)
            while result.done() is not True:
                result.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error creating snapshot %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())

    def delete_azure_netapp_snapshot(self):
        """
            Delete a snapshot for the given Azure NetApp Account
            :return: None
        """
        try:
            result = self.get_method('snapshots', 'delete')(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['volume_name'], snapshot_name=self.parameters['name'])
            while result.done() is not True:
                result.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error deleting snapshot %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())

    def exec_module(self, **kwargs):
        self.fail_when_import_errors(IMPORT_ERRORS, HAS_AZURE_MGMT_NETAPP)
        for key in list(self.module_arg_spec):
            self.parameters[key] = kwargs[key]
        current = self.get_azure_netapp_snapshot()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'create':
                self.create_azure_netapp_snapshot()
            elif cd_action == 'delete':
                self.delete_azure_netapp_snapshot()
        self.module.exit_json(changed=self.na_helper.changed)