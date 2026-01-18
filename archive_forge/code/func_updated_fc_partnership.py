from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def updated_fc_partnership(self, modification, restapi, cluster):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'chpartnership'
    if 'start' in modification:
        modification.pop('start')
        restapi.svc_run_command(cmd, {'start': True}, [cluster])
        self.changed = True
    if 'stop' in modification:
        modification.pop('stop')
        restapi.svc_run_command(cmd, {'stop': True}, [cluster])
        self.changed = True
    if modification:
        restapi.svc_run_command(cmd, modification, [cluster])
        self.changed = True