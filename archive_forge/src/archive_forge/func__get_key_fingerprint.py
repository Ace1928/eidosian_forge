from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _get_key_fingerprint(self, path):
    private_key_content = self.ssh_keygen.get_private_key(path, check_rc=True)[1]
    return PrivateKey.from_string(private_key_content).fingerprint