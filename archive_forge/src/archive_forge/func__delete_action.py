from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def _delete_action(self):
    cmd = ['rm', self.params['name']]
    if self.params['force']:
        cmd += ['--force']
    return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]