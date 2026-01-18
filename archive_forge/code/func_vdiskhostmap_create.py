from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def vdiskhostmap_create(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("creating vdiskhostmap '%s' '%s'", self.volname, self.host)
    cmd = 'mkvdiskhostmap'
    cmdopts = {'force': True}
    cmdopts['host'] = self.host
    cmdopts['scsi'] = self.scsi
    cmdargs = [self.volname]
    self.log('creating vdiskhostmap command %s opts %s args %s', cmd, cmdopts, cmdargs)
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.log('create vdiskhostmap result %s', result)
    if 'message' in result:
        self.changed = True
        self.log('create vdiskhostmap result message %s', result['message'])
    else:
        self.module.fail_json(msg='Failed to create vdiskhostmap.')