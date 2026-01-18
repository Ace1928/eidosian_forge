from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def update_cloud_backup(self):
    cmdopts = {}
    if self.enable_cloud_snapshot is True:
        cmdopts['backup'] = 'cloud'
        cmdopts['enable'] = True
    if self.enable_cloud_snapshot is False:
        cmdopts['backup'] = 'cloud'
        cmdopts['disable'] = True
    if self.cloud_account_name:
        cmdopts['account'] = self.cloud_account_name
    self.restapi.svc_run_command('chvdisk', cmdopts, [self.name])
    self.changed = True