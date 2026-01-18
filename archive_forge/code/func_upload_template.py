from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def upload_template(self, node, storage, content_type, realpath, timeout):
    stats = os.stat(realpath)
    if LooseVersion(self.proxmoxer_version) >= LooseVersion('1.2.0') and stats.st_size > 268435456 and (not HAS_REQUESTS_TOOLBELT):
        self.module.fail_json(msg="'requests_toolbelt' module is required to upload files larger than 256MB", exception=missing_required_lib('requests_toolbelt'))
    try:
        taskid = self.proxmox_api.nodes(node).storage(storage).upload.post(content=content_type, filename=open(realpath, 'rb'))
        return self.task_status(node, taskid, timeout)
    except Exception as e:
        self.module.fail_json(msg='Uploading template %s failed with error: %s' % (realpath, e))