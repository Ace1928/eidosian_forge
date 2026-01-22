from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def _get_key_format(self, key_format):
    result = 'SSH'
    if key_format == 'auto':
        ssh_version = self._get_ssh_version() or '7.8'
        if LooseVersion(ssh_version) < LooseVersion('7.8') and self.type != 'ed25519':
            result = 'PKCS1'
        if result == 'SSH' and (not HAS_OPENSSH_PRIVATE_FORMAT):
            self.module.fail_json(msg=missing_required_lib('cryptography >= 3.0', reason='to load/dump private keys in the default OpenSSH format for OpenSSH >= 7.8 ' + 'or for ed25519 keys'))
    else:
        result = key_format.upper()
    return result