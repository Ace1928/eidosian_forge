from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _is_cryptography_key_consistent(key, key_public_data, key_private_data, warn_func=None):
    if isinstance(key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
        backend = getattr(key, '_backend', None)
        if backend is not None:
            return bool(backend._lib.RSA_check_key(key._rsa_cdata))
    if isinstance(key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPrivateKey):
        result = _check_dsa_consistency(key_public_data, key_private_data)
        if result is not None:
            return result
        try:
            signature = key.sign(SIGNATURE_TEST_DATA, cryptography.hazmat.primitives.hashes.SHA256())
        except AttributeError:
            return None
        try:
            key.public_key().verify(signature, SIGNATURE_TEST_DATA, cryptography.hazmat.primitives.hashes.SHA256())
            return True
        except cryptography.exceptions.InvalidSignature:
            return False
    if isinstance(key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
        try:
            signature = key.sign(SIGNATURE_TEST_DATA, cryptography.hazmat.primitives.asymmetric.ec.ECDSA(cryptography.hazmat.primitives.hashes.SHA256()))
        except AttributeError:
            return None
        try:
            key.public_key().verify(signature, SIGNATURE_TEST_DATA, cryptography.hazmat.primitives.asymmetric.ec.ECDSA(cryptography.hazmat.primitives.hashes.SHA256()))
            return True
        except cryptography.exceptions.InvalidSignature:
            return False
    has_simple_sign_function = False
    if CRYPTOGRAPHY_HAS_ED25519 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey):
        has_simple_sign_function = True
    if CRYPTOGRAPHY_HAS_ED448 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey):
        has_simple_sign_function = True
    if has_simple_sign_function:
        signature = key.sign(SIGNATURE_TEST_DATA)
        try:
            key.public_key().verify(signature, SIGNATURE_TEST_DATA)
            return True
        except cryptography.exceptions.InvalidSignature:
            return False
    if warn_func is not None:
        warn_func('Cannot determine consistency for key of type %s' % type(key))
    return None