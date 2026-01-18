from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def update_safeguarded_copy_functionality(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chsystem'
    cmdopts = {'safeguardedcopysuspended': 'yes' if self.state == 'suspend' else 'no'}
    self.restapi.svc_run_command(cmd, cmdopts=cmdopts, cmdargs=None)
    self.log('Safeguarded copy functionality status changed: %s', self.state)
    self.changed = True