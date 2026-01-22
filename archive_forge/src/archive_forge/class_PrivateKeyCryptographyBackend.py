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
class PrivateKeyCryptographyBackend(PrivateKeyBackend):

    def _get_ec_class(self, ectype):
        ecclass = cryptography.hazmat.primitives.asymmetric.ec.__dict__.get(ectype)
        if ecclass is None:
            self.module.fail_json(msg='Your cryptography version does not support {0}'.format(ectype))
        return ecclass

    def _add_curve(self, name, ectype, deprecated=False):

        def create(size):
            ecclass = self._get_ec_class(ectype)
            return ecclass()

        def verify(privatekey):
            ecclass = self._get_ec_class(ectype)
            return isinstance(privatekey.private_numbers().public_numbers.curve, ecclass)
        self.curves[name] = {'create': create, 'verify': verify, 'deprecated': deprecated}

    def __init__(self, module):
        super(PrivateKeyCryptographyBackend, self).__init__(module=module, backend='cryptography')
        self.curves = dict()
        self._add_curve('secp224r1', 'SECP224R1')
        self._add_curve('secp256k1', 'SECP256K1')
        self._add_curve('secp256r1', 'SECP256R1')
        self._add_curve('secp384r1', 'SECP384R1')
        self._add_curve('secp521r1', 'SECP521R1')
        self._add_curve('secp192r1', 'SECP192R1', deprecated=True)
        self._add_curve('sect163k1', 'SECT163K1', deprecated=True)
        self._add_curve('sect163r2', 'SECT163R2', deprecated=True)
        self._add_curve('sect233k1', 'SECT233K1', deprecated=True)
        self._add_curve('sect233r1', 'SECT233R1', deprecated=True)
        self._add_curve('sect283k1', 'SECT283K1', deprecated=True)
        self._add_curve('sect283r1', 'SECT283R1', deprecated=True)
        self._add_curve('sect409k1', 'SECT409K1', deprecated=True)
        self._add_curve('sect409r1', 'SECT409R1', deprecated=True)
        self._add_curve('sect571k1', 'SECT571K1', deprecated=True)
        self._add_curve('sect571r1', 'SECT571R1', deprecated=True)
        self._add_curve('brainpoolP256r1', 'BrainpoolP256R1', deprecated=True)
        self._add_curve('brainpoolP384r1', 'BrainpoolP384R1', deprecated=True)
        self._add_curve('brainpoolP512r1', 'BrainpoolP512R1', deprecated=True)
        self.cryptography_backend = cryptography.hazmat.backends.default_backend()
        if not CRYPTOGRAPHY_HAS_X25519 and self.type == 'X25519':
            self.module.fail_json(msg='Your cryptography version does not support X25519')
        if not CRYPTOGRAPHY_HAS_X25519_FULL and self.type == 'X25519':
            self.module.fail_json(msg='Your cryptography version does not support X25519 serialization')
        if not CRYPTOGRAPHY_HAS_X448 and self.type == 'X448':
            self.module.fail_json(msg='Your cryptography version does not support X448')
        if not CRYPTOGRAPHY_HAS_ED25519 and self.type == 'Ed25519':
            self.module.fail_json(msg='Your cryptography version does not support Ed25519')
        if not CRYPTOGRAPHY_HAS_ED448 and self.type == 'Ed448':
            self.module.fail_json(msg='Your cryptography version does not support Ed448')

    def _get_wanted_format(self):
        if self.format not in ('auto', 'auto_ignore'):
            return self.format
        if self.type in ('X25519', 'X448', 'Ed25519', 'Ed448'):
            return 'pkcs8'
        else:
            return 'pkcs1'

    def generate_private_key(self):
        """(Re-)Generate private key."""
        try:
            if self.type == 'RSA':
                self.private_key = cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(public_exponent=65537, key_size=self.size, backend=self.cryptography_backend)
            if self.type == 'DSA':
                self.private_key = cryptography.hazmat.primitives.asymmetric.dsa.generate_private_key(key_size=self.size, backend=self.cryptography_backend)
            if CRYPTOGRAPHY_HAS_X25519_FULL and self.type == 'X25519':
                self.private_key = cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.generate()
            if CRYPTOGRAPHY_HAS_X448 and self.type == 'X448':
                self.private_key = cryptography.hazmat.primitives.asymmetric.x448.X448PrivateKey.generate()
            if CRYPTOGRAPHY_HAS_ED25519 and self.type == 'Ed25519':
                self.private_key = cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.generate()
            if CRYPTOGRAPHY_HAS_ED448 and self.type == 'Ed448':
                self.private_key = cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey.generate()
            if self.type == 'ECC' and self.curve in self.curves:
                if self.curves[self.curve]['deprecated']:
                    self.module.warn('Elliptic curves of type {0} should not be used for new keys!'.format(self.curve))
                self.private_key = cryptography.hazmat.primitives.asymmetric.ec.generate_private_key(curve=self.curves[self.curve]['create'](self.size), backend=self.cryptography_backend)
        except cryptography.exceptions.UnsupportedAlgorithm as dummy:
            self.module.fail_json(msg='Cryptography backend does not support the algorithm required for {0}'.format(self.type))

    def get_private_key_data(self):
        """Return bytes for self.private_key"""
        try:
            export_format = self._get_wanted_format()
            export_encoding = cryptography.hazmat.primitives.serialization.Encoding.PEM
            if export_format == 'pkcs1':
                export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.TraditionalOpenSSL
            elif export_format == 'pkcs8':
                export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.PKCS8
            elif export_format == 'raw':
                export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.Raw
                export_encoding = cryptography.hazmat.primitives.serialization.Encoding.Raw
        except AttributeError:
            self.module.fail_json(msg='Cryptography backend does not support the selected output format "{0}"'.format(self.format))
        encryption_algorithm = cryptography.hazmat.primitives.serialization.NoEncryption()
        if self.cipher and self.passphrase:
            if self.cipher == 'auto':
                encryption_algorithm = cryptography.hazmat.primitives.serialization.BestAvailableEncryption(to_bytes(self.passphrase))
            else:
                self.module.fail_json(msg='Cryptography backend can only use "auto" for cipher option.')
        try:
            return self.private_key.private_bytes(encoding=export_encoding, format=export_format, encryption_algorithm=encryption_algorithm)
        except ValueError as dummy:
            self.module.fail_json(msg='Cryptography backend cannot serialize the private key in the required format "{0}"'.format(self.format))
        except Exception as dummy:
            self.module.fail_json(msg='Error while serializing the private key in the required format "{0}"'.format(self.format), exception=traceback.format_exc())

    def _load_privatekey(self):
        data = self.existing_private_key_bytes
        try:
            format = identify_private_key_format(data)
            if format == 'raw':
                if len(data) == 56 and CRYPTOGRAPHY_HAS_X448:
                    return cryptography.hazmat.primitives.asymmetric.x448.X448PrivateKey.from_private_bytes(data)
                if len(data) == 57 and CRYPTOGRAPHY_HAS_ED448:
                    return cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey.from_private_bytes(data)
                if len(data) == 32:
                    if CRYPTOGRAPHY_HAS_X25519 and (self.type == 'X25519' or not CRYPTOGRAPHY_HAS_ED25519):
                        return cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.from_private_bytes(data)
                    if CRYPTOGRAPHY_HAS_ED25519 and (self.type == 'Ed25519' or not CRYPTOGRAPHY_HAS_X25519):
                        return cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.from_private_bytes(data)
                    if CRYPTOGRAPHY_HAS_X25519 and CRYPTOGRAPHY_HAS_ED25519:
                        try:
                            return cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.from_private_bytes(data)
                        except Exception:
                            return cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.from_private_bytes(data)
                raise PrivateKeyError('Cannot load raw key')
            else:
                return cryptography.hazmat.primitives.serialization.load_pem_private_key(data, None if self.passphrase is None else to_bytes(self.passphrase), backend=self.cryptography_backend)
        except Exception as e:
            raise PrivateKeyError(e)

    def _ensure_existing_private_key_loaded(self):
        if self.existing_private_key is None and self.has_existing():
            self.existing_private_key = self._load_privatekey()

    def _check_passphrase(self):
        try:
            format = identify_private_key_format(self.existing_private_key_bytes)
            if format == 'raw':
                self._load_privatekey()
                return self.passphrase is None
            else:
                return cryptography.hazmat.primitives.serialization.load_pem_private_key(self.existing_private_key_bytes, None if self.passphrase is None else to_bytes(self.passphrase), backend=self.cryptography_backend)
        except Exception as dummy:
            return False

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

    def _check_format(self):
        if self.format == 'auto_ignore':
            return True
        try:
            format = identify_private_key_format(self.existing_private_key_bytes)
            return format == self._get_wanted_format()
        except Exception as dummy:
            return False