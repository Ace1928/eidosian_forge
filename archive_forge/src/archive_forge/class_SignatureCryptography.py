from __future__ import absolute_import, division, print_function
import os
import traceback
import base64
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class SignatureCryptography(SignatureBase):

    def __init__(self, module, backend):
        super(SignatureCryptography, self).__init__(module, backend)

    def run(self):
        _padding = cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15()
        _hash = cryptography.hazmat.primitives.hashes.SHA256()
        result = dict()
        try:
            with open(self.path, 'rb') as f:
                _in = f.read()
            private_key = load_privatekey(path=self.privatekey_path, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
            signature = None
            if CRYPTOGRAPHY_HAS_DSA_SIGN:
                if isinstance(private_key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPrivateKey):
                    signature = private_key.sign(_in, _hash)
            if CRYPTOGRAPHY_HAS_EC_SIGN:
                if isinstance(private_key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
                    signature = private_key.sign(_in, cryptography.hazmat.primitives.asymmetric.ec.ECDSA(_hash))
            if CRYPTOGRAPHY_HAS_ED25519_SIGN:
                if isinstance(private_key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey):
                    signature = private_key.sign(_in)
            if CRYPTOGRAPHY_HAS_ED448_SIGN:
                if isinstance(private_key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey):
                    signature = private_key.sign(_in)
            if CRYPTOGRAPHY_HAS_RSA_SIGN:
                if isinstance(private_key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
                    signature = private_key.sign(_in, _padding, _hash)
            if signature is None:
                self.module.fail_json(msg='Unsupported key type. Your cryptography version is {0}'.format(CRYPTOGRAPHY_VERSION))
            result['signature'] = base64.b64encode(signature)
            return result
        except Exception as e:
            raise OpenSSLObjectError(e)