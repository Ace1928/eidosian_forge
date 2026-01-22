from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
class AwsCvsNetappSnapshot(object):
    """
    Contains methods to parse arguments,
    derive details of AWS_CVS objects
    and send requests to AWS CVS via
    the restApi
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check paramenters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.aws_cvs_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=True, choices=['present', 'absent']), region=dict(required=True, type='str'), name=dict(required=True, type='str'), from_name=dict(required=False, type='str'), fileSystemId=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['fileSystemId'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = AwsCvsRestAPI(self.module)
        self.data = {}
        for key in self.parameters.keys():
            self.data[key] = self.parameters[key]

    def get_snapshot_id(self, name):
        list_snapshots, error = self.rest_api.get('Snapshots')
        if error:
            self.module.fail_json(msg=error)
        for snapshot in list_snapshots:
            if snapshot['name'] == name:
                return snapshot['snapshotId']
        return None

    def get_filesystem_id(self):
        list_filesystem, error = self.rest_api.get('FileSystems')
        if error:
            self.module.fail_json(msg=error)
        for filesystem in list_filesystem:
            if filesystem['fileSystemId'] == self.parameters['fileSystemId']:
                return filesystem['fileSystemId']
            elif filesystem['creationToken'] == self.parameters['fileSystemId']:
                return filesystem['fileSystemId']
        return None

    def create_snapshot(self):
        api = 'Snapshots'
        dummy, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def rename_snapshot(self, snapshot_id):
        api = 'Snapshots/' + snapshot_id
        dummy, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def delete_snapshot(self, snapshot_id):
        api = 'Snapshots/' + snapshot_id
        dummy, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        self.snapshot_id = self.get_snapshot_id(self.data['name'])
        if self.snapshot_id is None and 'fileSystemId' in self.data:
            self.filesystem_id = self.get_filesystem_id()
            self.data['fileSystemId'] = self.filesystem_id
            if self.filesystem_id is None:
                self.module.fail_json(msg='Error: Specified filesystem id %s does not exist ' % self.data['fileSystemId'])
        cd_action = self.na_helper.get_cd_action(self.snapshot_id, self.data)
        result_message = ''
        if self.na_helper.changed:
            if self.module.check_mode:
                result_message = 'Check mode, skipping changes'
            elif cd_action == 'delete':
                self.delete_snapshot(self.snapshot_id)
                result_message = 'Snapshot Deleted'
            elif cd_action == 'create':
                if 'from_name' in self.data:
                    snapshot_id = self.get_snapshot_id(self.data['from_name'])
                    if snapshot_id is not None:
                        self.rename_snapshot(snapshot_id)
                        result_message = 'Snapshot Updated'
                    else:
                        self.module.fail_json(msg='Resource does not exist : %s' % self.data['from_name'])
                else:
                    self.create_snapshot()
                    result_message = 'Snapshot Created'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message)