from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def update_aws_account(self, updates):
    if self.module.check_mode:
        self.changed = True
        return
    name = self.old_name if self.old_name else self.name
    self.restapi.svc_run_command('chcloudaccountawss3', updates, cmdargs=[name], timeout=20)
    self.changed = True