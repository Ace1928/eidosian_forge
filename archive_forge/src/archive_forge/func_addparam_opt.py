from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
def addparam_opt(self, c):
    for opt in self.params['opt'].items():
        if opt[1] is not None:
            c += ['--opt', b'='.join([to_bytes(k, errors='surrogate_or_strict') for k in opt])]
    return c