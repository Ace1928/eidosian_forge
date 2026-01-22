from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
class IBMSVCVG(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), state=dict(type='str', required=True, choices=['absent', 'present']), ownershipgroup=dict(type='str', required=False), noownershipgroup=dict(type='bool', required=False), safeguardpolicyname=dict(type='str', required=False), nosafeguardpolicy=dict(type='bool', required=False), policystarttime=dict(type='str', required=False), snapshotpolicy=dict(type='str', required=False), nosnapshotpolicy=dict(type='bool', required=False), snapshotpolicysuspended=dict(type='str', choices=['yes', 'no']), type=dict(type='str', choices=['clone', 'thinclone']), snapshot=dict(type='str'), fromsourcegroup=dict(type='str'), pool=dict(type='str'), iogrp=dict(type='str'), safeguarded=dict(type='bool', default=False), ignoreuserfcmaps=dict(type='str', choices=['yes', 'no']), replicationpolicy=dict(type='str'), noreplicationpolicy=dict(type='bool')))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.name = self.module.params['name']
        self.state = self.module.params['state']
        self.ownershipgroup = self.module.params.get('ownershipgroup', '')
        self.noownershipgroup = self.module.params.get('noownershipgroup', False)
        self.policystarttime = self.module.params.get('policystarttime', '')
        self.snapshotpolicy = self.module.params.get('snapshotpolicy', '')
        self.nosnapshotpolicy = self.module.params.get('nosnapshotpolicy', False)
        self.snapshotpolicysuspended = self.module.params.get('snapshotpolicysuspended', '')
        self.type = self.module.params.get('type', '')
        self.snapshot = self.module.params.get('snapshot', '')
        self.fromsourcegroup = self.module.params.get('fromsourcegroup', '')
        self.pool = self.module.params.get('pool', '')
        self.iogrp = self.module.params.get('iogrp', '')
        self.safeguardpolicyname = self.module.params.get('safeguardpolicyname', '')
        self.nosafeguardpolicy = self.module.params.get('nosafeguardpolicy', False)
        self.safeguarded = self.module.params.get('safeguarded', False)
        self.ignoreuserfcmaps = self.module.params.get('ignoreuserfcmaps', '')
        self.replicationpolicy = self.module.params.get('replicationpolicy', '')
        self.noreplicationpolicy = self.module.params.get('noreplicationpolicy', False)
        self.parentuid = None
        self.changed = False
        self.msg = ''
        self.basic_checks()
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')
        if self.state == 'present':
            if self.policystarttime:
                if not self.snapshotpolicy and (not self.safeguardpolicyname):
                    self.module.fail_json(msg='Either `snapshotpolicy` or `safeguardpolicyname` should be passed along with `policystarttime`.')
            if self.safeguarded:
                if not self.snapshotpolicy:
                    self.module.fail_json(msg='Parameter `safeguarded` should be passed along with `snapshotpolicy`')
        else:
            unwanted = ('ownershipgroup', 'noownershipgroup', 'safeguardpolicyname', 'nosafeguardpolicy', 'snapshotpolicy', 'nosnapshotpolicy', 'policystarttime', 'type', 'fromsourcegroup', 'pool', 'iogrp', 'safeguarded', 'ignoreuserfcmaps', 'replicationpolicy', 'noreplicationpolicy')
            param_exists = ', '.join((param for param in unwanted if getattr(self, param)))
            if param_exists:
                self.module.fail_json(msg='State=absent but following parameters exists: {0}'.format(param_exists))

    def create_validation(self):
        mutually_exclusive = (('ownershipgroup', 'safeguardpolicyname'), ('ownershipgroup', 'snapshotpolicy'), ('ownershipgroup', 'policystarttime'), ('snapshotpolicy', 'safeguardpolicyname'), ('replicationpolicy', 'noreplicationpolicy'))
        for param1, param2 in mutually_exclusive:
            if getattr(self, param1) and getattr(self, param2):
                self.module.fail_json(msg='Mutually exclusive parameters: {0}, {1}'.format(param1, param2))
        unsupported = ('nosafeguardpolicy', 'noownershipgroup', 'nosnapshotpolicy', 'snapshotpolicysuspended', 'noreplicationpolicy')
        unsupported_exists = ', '.join((field for field in unsupported if getattr(self, field)))
        if unsupported_exists:
            self.module.fail_json(msg='Following paramters not supported during creation scenario: {0}'.format(unsupported_exists))
        if self.type and (not self.snapshot):
            self.module.fail_json(msg='type={0} but following parameter is missing: snapshot'.format(self.type))

    def update_validation(self, data):
        mutually_exclusive = (('ownershipgroup', 'noownershipgroup'), ('safeguardpolicyname', 'nosafeguardpolicy'), ('ownershipgroup', 'safeguardpolicyname'), ('ownershipgroup', 'snapshotpolicy'), ('ownershipgroup', 'policystarttime'), ('nosafeguardpolicy', 'nosnapshotpolicy'), ('snapshotpolicy', 'nosnapshotpolicy'), ('snapshotpolicy', 'safeguardpolicyname'), ('replicationpolicy', 'noreplicationpolicy'))
        for param1, param2 in mutually_exclusive:
            if getattr(self, param1) and getattr(self, param2):
                self.module.fail_json(msg='Mutually exclusive parameters: {0}, {1}'.format(param1, param2))
        unsupported_maps = (('type', data.get('volume_group_type', '')), ('snapshot', data.get('source_snapshot', '')), ('fromsourcegroup', data.get('source_volume_group_name', '')))
        unsupported = (fields[0] for fields in unsupported_maps if getattr(self, fields[0]) and getattr(self, fields[0]) != fields[1])
        unsupported_exists = ', '.join(unsupported)
        if unsupported_exists:
            self.module.fail_json(msg='Following paramters not supported during update: {0}'.format(unsupported_exists))

    def get_existing_vg(self):
        merged_result = {}
        data = self.restapi.svc_obj_info(cmd='lsvolumegroup', cmdopts=None, cmdargs=['-gui', self.name])
        if isinstance(data, list):
            for d in data:
                merged_result.update(d)
        else:
            merged_result = data
        if merged_result and (self.snapshotpolicy and self.policystarttime or self.snapshotpolicysuspended):
            SP_data = self.restapi.svc_obj_info(cmd='lsvolumegroupsnapshotpolicy', cmdopts=None, cmdargs=[self.name])
            merged_result['snapshot_policy_start_time'] = SP_data['snapshot_policy_start_time']
            merged_result['snapshot_policy_suspended'] = SP_data['snapshot_policy_suspended']
        return merged_result

    def set_parentuid(self):
        if self.snapshot and (not self.fromsourcegroup):
            cmdopts = {'filtervalue': 'snapshot_name={0}'.format(self.snapshot)}
            data = self.restapi.svc_obj_info(cmd='lsvolumesnapshot', cmdopts=cmdopts, cmdargs=None)
            try:
                result = next(filter(lambda obj: obj['volume_group_name'] == '', data))
            except StopIteration:
                self.module.fail_json(msg='Orphan Snapshot ({0}) does not exists for the given name'.format(self.snapshot))
            else:
                self.parentuid = result['parent_uid']

    def vg_probe(self, data):
        self.update_validation(data)
        params_mapping = (('ownershipgroup', data.get('owner_name', '')), ('ignoreuserfcmaps', data.get('ignore_user_flash_copy_maps', '')), ('replicationpolicy', data.get('replication_policy_name', '')), ('noownershipgroup', not bool(data.get('owner_name', ''))), ('nosafeguardpolicy', not bool(data.get('safeguarded_policy_name', ''))), ('nosnapshotpolicy', not bool(data.get('snapshot_policy_name', ''))), ('noreplicationpolicy', not bool(data.get('replication_policy_name', ''))))
        props = dict(((k, getattr(self, k)) for k, v in params_mapping if getattr(self, k) and getattr(self, k) != v))
        if self.safeguardpolicyname and self.safeguardpolicyname != data.get('safeguarded_policy_name', ''):
            props['safeguardedpolicy'] = self.safeguardpolicyname
            if self.policystarttime:
                props['policystarttime'] = self.policystarttime
        elif self.safeguardpolicyname:
            if self.policystarttime and self.policystarttime + '00' != data.get('safeguarded_policy_start_time', ''):
                props['safeguardedpolicy'] = self.safeguardpolicyname
                props['policystarttime'] = self.policystarttime
        elif self.snapshotpolicy and self.snapshotpolicy != data.get('snapshot_policy_name', ''):
            props['snapshotpolicy'] = self.snapshotpolicy
            props['safeguarded'] = self.safeguarded
            if self.policystarttime:
                props['policystarttime'] = self.policystarttime
        elif self.snapshotpolicy:
            if self.policystarttime and self.policystarttime + '00' != data.get('snapshot_policy_start_time', ''):
                props['snapshotpolicy'] = self.snapshotpolicy
                props['policystarttime'] = self.policystarttime
            if self.safeguarded not in ('', None) and self.safeguarded != strtobool(data.get('snapshot_policy_safeguarded', 0)):
                props['snapshotpolicy'] = self.snapshotpolicy
                props['safeguarded'] = self.safeguarded
        if self.snapshotpolicysuspended and self.snapshotpolicysuspended != data.get('snapshot_policy_suspended', ''):
            props['snapshotpolicysuspended'] = self.snapshotpolicysuspended
        self.log('volumegroup props = %s', props)
        return props

    def vg_create(self):
        self.create_validation()
        if self.module.check_mode:
            self.changed = True
            return
        self.log("creating volume group '%s'", self.name)
        cmd = 'mkvolumegroup'
        cmdopts = {'name': self.name, 'safeguarded': self.safeguarded}
        if self.type:
            optional_params = ('type', 'snapshot', 'pool')
            cmdopts.update(dict(((param, getattr(self, param)) for param in optional_params if getattr(self, param))))
            if self.iogrp:
                cmdopts['iogroup'] = self.iogrp
            self.set_parentuid()
            if self.parentuid:
                cmdopts['fromsourceuid'] = self.parentuid
            else:
                cmdopts['fromsourcegroup'] = self.fromsourcegroup
        if self.ignoreuserfcmaps:
            if self.ignoreuserfcmaps == 'yes':
                cmdopts['ignoreuserfcmaps'] = True
            else:
                cmdopts['ignoreuserfcmaps'] = False
        if self.replicationpolicy:
            cmdopts['replicationpolicy'] = self.replicationpolicy
        if self.ownershipgroup:
            cmdopts['ownershipgroup'] = self.ownershipgroup
        elif self.safeguardpolicyname:
            cmdopts['safeguardedpolicy'] = self.safeguardpolicyname
            if self.policystarttime:
                cmdopts['policystarttime'] = self.policystarttime
        elif self.snapshotpolicy:
            cmdopts['snapshotpolicy'] = self.snapshotpolicy
            if self.policystarttime:
                cmdopts['policystarttime'] = self.policystarttime
        self.log("creating volumegroup '%s'", cmdopts)
        result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('create volume group result %s', result)
        self.changed = True

    def vg_update(self, modify):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("updating volume group '%s' ", self.name)
        cmdargs = [self.name]
        try:
            del modify['snapshotpolicysuspended']
        except KeyError:
            self.log('snapshotpolicysuspended modification not reqiured!!')
        else:
            cmd = 'chvolumegroupsnapshotpolicy'
            cmdopts = {'snapshotpolicysuspended': self.snapshotpolicysuspended}
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        cmd = 'chvolumegroup'
        unmaps = ('noownershipgroup', 'nosafeguardpolicy', 'nosnapshotpolicy', 'noreplicationpolicy')
        for field in unmaps:
            cmdopts = {}
            if field == 'nosafeguardpolicy' and field in modify:
                cmdopts['nosafeguardedpolicy'] = modify.pop('nosafeguardpolicy')
                self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
            elif field in modify:
                cmdopts[field] = modify.pop(field)
                self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        if modify:
            cmdopts = modify
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def vg_delete(self):
        if self.module.check_mode:
            self.changed = True
            return
        self.log("deleting volume group '%s'", self.name)
        cmd = 'rmvolumegroup'
        cmdopts = None
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True

    def apply(self):
        vg_data = self.get_existing_vg()
        if vg_data:
            if self.state == 'present':
                modify = self.vg_probe(vg_data)
                if modify:
                    self.vg_update(modify)
                    self.msg = 'volume group [%s] has been modified.' % self.name
                else:
                    self.msg = 'No Modifications detected, Volume group already exists.'
            else:
                self.vg_delete()
                self.msg = 'volume group [%s] has been deleted.' % self.name
        elif self.state == 'absent':
            self.msg = 'Volume group [%s] does not exist.' % self.name
        else:
            self.vg_create()
            self.msg = 'volume group [%s] has been created.' % self.name
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(msg=self.msg, changed=self.changed)