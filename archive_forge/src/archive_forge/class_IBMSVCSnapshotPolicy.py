from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVCSnapshotPolicy:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=True, choices=['present', 'absent', 'suspend', 'resume']), name=dict(type='str'), backupunit=dict(type='str', choices=['minute', 'hour', 'day', 'week', 'month']), backupinterval=dict(type='str'), backupstarttime=dict(type='str'), retentiondays=dict(type='str'), removefromvolumegroups=dict(type='bool')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.backupunit = self.module.params.get('backupunit', '')
        self.backupinterval = self.module.params.get('backupinterval', '')
        self.backupstarttime = self.module.params.get('backupstarttime', '')
        self.retentiondays = self.module.params.get('retentiondays', '')
        self.removefromvolumegroups = self.module.params.get('removefromvolumegroups', False)
        self.basic_checks()
        self.snapshot_policy_details = None
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if self.state == 'present':
            fields = ['name', 'backupinterval', 'backupstarttime', 'retentiondays', 'backupunit']
            exists = list(filter(lambda x: not getattr(self, x), fields))
            if any(exists):
                self.module.fail_json(msg='State is present but following parameters are missing: {0}'.format(', '.join(exists)))
            if self.removefromvolumegroups:
                self.module.fail_json(msg='`removefromvolumegroups` parameter is not supported when state=present')
        elif self.state == 'absent':
            if not self.name:
                self.module.fail_json(msg='Missing mandatory parameter: name')
            fields = ['backupinterval', 'backupstarttime', 'retentiondays', 'backupunit']
            exists = list(filter(lambda x: getattr(self, x) or getattr(self, x) == '', fields))
            if any(exists):
                self.module.fail_json(msg='{0} should not be passed when state=absent'.format(', '.join(exists)))
        elif self.state in ['suspend', 'resume']:
            fields = ['name', 'backupinterval', 'backupstarttime', 'retentiondays', 'backupunit']
            exists = list(filter(lambda x: getattr(self, x) or getattr(self, x) == '', fields))
            if any(exists):
                self.module.fail_json(msg='{0} should not be passed when state={1}'.format(', '.join(exists), self.state))

    def policy_exists(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lssnapshotschedule', cmdopts=None, cmdargs=[self.name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        self.snapshot_policy_details = merged_result
        return merged_result

    def create_snapshot_policy(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mksnapshotpolicy'
        cmdopts = {'name': self.name, 'backupstarttime': self.backupstarttime, 'backupinterval': self.backupinterval, 'backupunit': self.backupunit, 'retentiondays': self.retentiondays}
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('Snapshot policy (%s) created', self.name)
        self.changed = True

    def snapshot_policy_probe(self):
        field_mappings = (('backupinterval', self.snapshot_policy_details['backup_interval']), ('backupstarttime', self.snapshot_policy_details['backup_start_time']), ('retentiondays', self.snapshot_policy_details['retention_days']), ('backupunit', self.snapshot_policy_details['backup_unit']))
        updates = []
        for field, existing_value in field_mappings:
            if field == 'backupstarttime':
                updates.append(existing_value != '{0}00'.format(getattr(self, field)))
            else:
                updates.append(existing_value != getattr(self, field))
        return updates

    def delete_snapshot_policy(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmsnapshotpolicy'
        cmdargs = [self.name]
        cmdopts = None
        if self.removefromvolumegroups:
            cmdopts = {'removefromvolumegroups': True}
        self.restapi.svc_run_command(cmd, cmdopts=cmdopts, cmdargs=cmdargs)
        self.log('Snapshot policy (%s) deleted', self.name)
        self.changed = True

    def update_snapshot_scheduler(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'chsystem'
        cmdopts = {'snapshotpolicysuspended': 'yes' if self.state == 'suspend' else 'no'}
        self.restapi.svc_run_command(cmd, cmdopts=cmdopts, cmdargs=None)
        self.log('Snapshot scheduler status changed: %s', self.state)
        self.changed = True

    def apply(self):
        if self.state in ['resume', 'suspend']:
            self.update_snapshot_scheduler()
            self.msg = 'Snapshot scheduler {0}ed'.format(self.state.rstrip('e'))
        elif self.policy_exists():
            if self.state == 'present':
                modifications = self.snapshot_policy_probe()
                if any(modifications):
                    self.msg = 'Policy modification is not supported in ansible. Please delete and recreate new policy.'
                else:
                    self.msg = 'Snapshot policy ({0}) already exists. No modifications done.'.format(self.name)
            else:
                self.delete_snapshot_policy()
                self.msg = 'Snapshot policy ({0}) deleted.'.format(self.name)
        elif self.state == 'absent':
            self.msg = 'Snapshot policy ({0}) does not exist. No modifications done.'.format(self.name)
        else:
            self.create_snapshot_policy()
            self.msg = 'Snapshot policy ({0}) created.'.format(self.name)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)