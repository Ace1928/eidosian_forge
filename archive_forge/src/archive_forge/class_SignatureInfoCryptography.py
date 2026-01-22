from __future__ import absolute_import, division, print_function
import os
import traceback
import base64
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class SignatureInfoCryptography(SignatureInfoBase):

    def __init__(self, module, backend):
        super(SignatureInfoCryptography, self).__init__(module, backend)

    def run(self):
        _padding = cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15()
        _hash = cryptography.hazmat.primitives.hashes.SHA256()
        result = dict()
        try:
            with open(self.path, 'rb') as f:
                _in = f.read()
            _signature = base64.b64decode(self.signature)
            certificate = load_certificate(path=self.certificate_path, content=self.certificate_content, backend=self.backend)
            public_key = certificate.public_key()
            verified = False
            valid = False
            if CRYPTOGRAPHY_HAS_DSA_SIGN:
                try:
                    if isinstance(public_key, cryptography.hazmat.primitives.asymmetric.dsa.DSAPublicKey):
                        public_key.verify(_signature, _in, _hash)
                        verified = True
                        valid = True
                except cryptography.exceptions.InvalidSignature:
                    verified = True
                    valid = False
            if CRYPTOGRAPHY_HAS_EC_SIGN:
                try:
                    if isinstance(public_key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePublicKey):
                        public_key.verify(_signature, _in, cryptography.hazmat.primitives.asymmetric.ec.ECDSA(_hash))
                        verified = True
                        valid = True
                except cryptography.exceptions.InvalidSignature:
                    verified = True
                    valid = False
            if CRYPTOGRAPHY_HAS_ED25519_SIGN:
                try:
                    if isinstance(public_key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PublicKey):
                        public_key.verify(_signature, _in)
                        verified = True
                        valid = True
                except cryptography.exceptions.InvalidSignature:
                    verified = True
                    valid = False
            if CRYPTOGRAPHY_HAS_ED448_SIGN:
                try:
                    if isinstance(public_key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PublicKey):
                        public_key.verify(_signature, _in)
                        verified = True
                        valid = True
                except cryptography.exceptions.InvalidSignature:
                    verified = True
                    valid = False
            if CRYPTOGRAPHY_HAS_RSA_SIGN:
                try:
                    if isinstance(public_key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey):
                        public_key.verify(_signature, _in, _padding, _hash)
                        verified = True
                        valid = True
                except cryptography.exceptions.InvalidSignature:
                    verified = True
                    valid = False
            if not verified:
                self.module.fail_json(msg='Unsupported key type. Your cryptography version is {0}'.format(CRYPTOGRAPHY_VERSION))
            result['valid'] = valid
            return result
        except Exception as e:
            raise OpenSSLObjectError(e)