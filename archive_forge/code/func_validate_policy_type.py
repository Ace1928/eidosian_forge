from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def validate_policy_type(self):
    if self.parameters['state'] != 'present':
        return
    self.validate_async_options()
    if 'policy_type' in self.parameters:
        if self.use_rest:
            if self.parameters['policy_type'] == 'vault':
                self.parameters['policy_type'] = 'async'
                self.parameters['create_snapshot_on_source'] = False
                self.module.warn("policy type changed to 'async' with 'create_snapshot_on_source' set to False")
            if self.parameters['policy_type'] == 'async_mirror':
                self.parameters['policy_type'] = 'async'
                if 'copy_latest_source_snapshot' not in self.parameters or 'copy_all_source_snapshots' not in self.parameters:
                    self.parameters['copy_latest_source_snapshot'] = True
                    self.module.warn("policy type changed to 'async' with copy_latest_source_snapshot set to True.  Use async with copy_latest_source_snapshot or copy_all_source_snapshots for async-mirror")
            if 'copy_all_source_snapshots' in self.parameters or 'copy_latest_source_snapshot' in self.parameters:
                if 'snapmirror_label' in self.parameters or 'keep' in self.parameters or 'prefix' in self.parameters or ('schedule' in self.parameters):
                    self.module.fail_json(msg='Error: Retention properties cannot be specified along with copy_all_source_snapshots or copy_latest_source_snapshot properties')
            if 'create_snapshot_on_source' in self.parameters:
                if 'snapmirror_label' not in self.parameters or 'keep' not in self.parameters:
                    self.module.fail_json(msg='Error: The properties snapmirror_label and keep must be specified with create_snapshot_on_source set to false')
            if self.parameters['policy_type'] == 'mirror_vault':
                self.parameters['policy_type'] = 'async'
            if self.parameters['policy_type'] in ('sync_mirror', 'strict_sync_mirror'):
                self.parameters['sync_type'] = 'sync' if self.parameters['policy_type'] == 'sync_mirror' else 'strict_sync'
                self.parameters['policy_type'] = 'sync'
            if self.parameters['policy_type'] != 'sync' and 'sync_type' in self.parameters:
                self.module.fail_json(msg="Error: 'sync_type' is only applicable for sync policy_type")
        elif self.parameters['policy_type'] in ['async', 'sync']:
            self.module.fail_json(msg='Error: The policy types async and sync are not supported in ZAPI.')