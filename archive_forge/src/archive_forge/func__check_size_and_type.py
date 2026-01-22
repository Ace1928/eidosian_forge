from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _check_size_and_type(self):
    if isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
        return self.type == 'RSA' and self.size == self.existing_private_key.key_size
    if isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPrivateKey):
        return self.type == 'DSA' and self.size == self.existing_private_key.key_size
    if CRYPTOGRAPHY_HAS_X25519 and isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey):
        return self.type == 'X25519'
    if CRYPTOGRAPHY_HAS_X448 and isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.x448.X448PrivateKey):
        return self.type == 'X448'
    if CRYPTOGRAPHY_HAS_ED25519 and isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey):
        return self.type == 'Ed25519'
    if CRYPTOGRAPHY_HAS_ED448 and isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey):
        return self.type == 'Ed448'
    if isinstance(self.existing_private_key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
        if self.type != 'ECC':
            return False
        if self.curve not in self.curves:
            return False
        return self.curves[self.curve]['verify'](self.existing_private_key)
    return False