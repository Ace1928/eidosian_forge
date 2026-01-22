from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
class IBMSVSwitchReplication:

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=True), mode=dict(type='str', choices=['independent', 'production'], required=True)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.name = self.module.params['name']
        self.mode = self.module.params['mode']
        self.basic_checks()
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.changed = False
        self.msg = ''
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=self.log_path, token=self.module.params['token'])

    def basic_checks(self):
        if not self.name:
            self.module.fail_json(msg='Missing mandatory parameter: name')

    def get_volumegroup_info(self):
        return self.restapi.svc_obj_info('lsvolumegroup', None, [self.name])

    def change_vg_mode(self):
        cmd = 'chvolumegroupreplication'
        cmdopts = {}
        cmdopts['mode'] = self.mode
        self.log('Changing replicaiton direction.. Command %s opts %s', cmd, cmdopts)
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])

    def apply(self):
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        elif self.get_volumegroup_info():
            self.change_vg_mode()
            self.changed = True
            self.msg = 'Replication direction on volume group [%s] has been modified.' % self.name
        else:
            self.module.fail_json(msg='Volume group does not exist: [%s]' % self.name)
        self.module.exit_json(changed=self.changed, msg=self.msg)