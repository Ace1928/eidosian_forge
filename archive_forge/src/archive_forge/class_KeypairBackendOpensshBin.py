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
class KeypairBackendOpensshBin(KeypairBackend):

    def __init__(self, module):
        super(KeypairBackendOpensshBin, self).__init__(module)
        if self.module.params['private_key_format'] != 'auto':
            self.module.fail_json(msg="'auto' is the only valid option for " + "'private_key_format' when 'backend' is not 'cryptography'")
        self.ssh_keygen = KeygenCommand(self.module)

    def _generate_keypair(self, private_key_path):
        self.ssh_keygen.generate_keypair(private_key_path, self.size, self.type, self.comment, check_rc=True)

    def _get_private_key(self):
        rc, private_key_content, err = self.ssh_keygen.get_private_key(self.private_key_path, check_rc=False)
        if rc != 0:
            raise ValueError(err)
        return PrivateKey.from_string(private_key_content)

    def _get_public_key(self):
        public_key_content = self.ssh_keygen.get_matching_public_key(self.private_key_path, check_rc=True)[1]
        return PublicKey.from_string(public_key_content)

    def _private_key_readable(self):
        rc, stdout, stderr = self.ssh_keygen.get_matching_public_key(self.private_key_path, check_rc=False)
        return not (rc == 255 or any_in(stderr, 'is not a public key file', 'incorrect passphrase', 'load failed'))

    def _update_comment(self):
        try:
            ssh_version = self._get_ssh_version() or '7.8'
            force_new_format = LooseVersion('6.5') <= LooseVersion(ssh_version) < LooseVersion('7.8')
            self.ssh_keygen.update_comment(self.private_key_path, self.comment, force_new_format=force_new_format, check_rc=True)
        except (IOError, OSError) as e:
            self.module.fail_json(msg=to_native(e))

    def _private_key_valid_backend(self):
        return True