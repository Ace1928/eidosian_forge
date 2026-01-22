from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import read_file
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
class CryptographyBackend(CryptoBackend):

    def __init__(self, module):
        super(CryptographyBackend, self).__init__(module)

    def parse_key(self, key_file=None, key_content=None, passphrase=None):
        """
        Parses an RSA or Elliptic Curve key file in PEM format and returns key_data.
        Raises KeyParsingError in case of errors.
        """
        if key_content is None:
            key_content = read_file(key_file)
        else:
            key_content = to_bytes(key_content)
        try:
            key = cryptography.hazmat.primitives.serialization.load_pem_private_key(key_content, password=to_bytes(passphrase) if passphrase is not None else None, backend=_cryptography_backend)
        except Exception as e:
            raise KeyParsingError('error while loading key: {0}'.format(e))
        if isinstance(key, cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
            pk = key.public_key().public_numbers()
            return {'key_obj': key, 'type': 'rsa', 'alg': 'RS256', 'jwk': {'kty': 'RSA', 'e': nopad_b64(convert_int_to_bytes(pk.e)), 'n': nopad_b64(convert_int_to_bytes(pk.n))}, 'hash': 'sha256'}
        elif isinstance(key, cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
            pk = key.public_key().public_numbers()
            if pk.curve.name == 'secp256r1':
                bits = 256
                alg = 'ES256'
                hashalg = 'sha256'
                point_size = 32
                curve = 'P-256'
            elif pk.curve.name == 'secp384r1':
                bits = 384
                alg = 'ES384'
                hashalg = 'sha384'
                point_size = 48
                curve = 'P-384'
            elif pk.curve.name == 'secp521r1':
                bits = 521
                alg = 'ES512'
                hashalg = 'sha512'
                point_size = 66
                curve = 'P-521'
            else:
                raise KeyParsingError('unknown elliptic curve: {0}'.format(pk.curve.name))
            num_bytes = (bits + 7) // 8
            return {'key_obj': key, 'type': 'ec', 'alg': alg, 'jwk': {'kty': 'EC', 'crv': curve, 'x': nopad_b64(convert_int_to_bytes(pk.x, count=num_bytes)), 'y': nopad_b64(convert_int_to_bytes(pk.y, count=num_bytes))}, 'hash': hashalg, 'point_size': point_size}
        else:
            raise KeyParsingError('unknown key type "{0}"'.format(type(key)))

    def sign(self, payload64, protected64, key_data):
        sign_payload = '{0}.{1}'.format(protected64, payload64).encode('utf8')
        if 'mac_obj' in key_data:
            mac = key_data['mac_obj']()
            mac.update(sign_payload)
            signature = mac.finalize()
        elif isinstance(key_data['key_obj'], cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
            padding = cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15()
            hashalg = cryptography.hazmat.primitives.hashes.SHA256
            signature = key_data['key_obj'].sign(sign_payload, padding, hashalg())
        elif isinstance(key_data['key_obj'], cryptography.hazmat.primitives.asymmetric.ec.EllipticCurvePrivateKey):
            if key_data['hash'] == 'sha256':
                hashalg = cryptography.hazmat.primitives.hashes.SHA256
            elif key_data['hash'] == 'sha384':
                hashalg = cryptography.hazmat.primitives.hashes.SHA384
            elif key_data['hash'] == 'sha512':
                hashalg = cryptography.hazmat.primitives.hashes.SHA512
            ecdsa = cryptography.hazmat.primitives.asymmetric.ec.ECDSA(hashalg())
            r, s = cryptography.hazmat.primitives.asymmetric.utils.decode_dss_signature(key_data['key_obj'].sign(sign_payload, ecdsa))
            rr = convert_int_to_hex(r, 2 * key_data['point_size'])
            ss = convert_int_to_hex(s, 2 * key_data['point_size'])
            signature = binascii.unhexlify(rr) + binascii.unhexlify(ss)
        return {'protected': protected64, 'payload': payload64, 'signature': nopad_b64(signature)}

    def create_mac_key(self, alg, key):
        """Create a MAC key."""
        if alg == 'HS256':
            hashalg = cryptography.hazmat.primitives.hashes.SHA256
            hashbytes = 32
        elif alg == 'HS384':
            hashalg = cryptography.hazmat.primitives.hashes.SHA384
            hashbytes = 48
        elif alg == 'HS512':
            hashalg = cryptography.hazmat.primitives.hashes.SHA512
            hashbytes = 64
        else:
            raise BackendException('Unsupported MAC key algorithm for cryptography backend: {0}'.format(alg))
        key_bytes = base64.urlsafe_b64decode(key)
        if len(key_bytes) < hashbytes:
            raise BackendException('{0} key must be at least {1} bytes long (after Base64 decoding)'.format(alg, hashbytes))
        return {'mac_obj': lambda: cryptography.hazmat.primitives.hmac.HMAC(key_bytes, hashalg(), _cryptography_backend), 'type': 'hmac', 'alg': alg, 'jwk': {'kty': 'oct', 'k': key}}

    def get_csr_identifiers(self, csr_filename=None, csr_content=None):
        """
        Return a set of requested identifiers (CN and SANs) for the CSR.
        Each identifier is a pair (type, identifier), where type is either
        'dns' or 'ip'.
        """
        identifiers = set([])
        if csr_content is None:
            csr_content = read_file(csr_filename)
        else:
            csr_content = to_bytes(csr_content)
        csr = cryptography.x509.load_pem_x509_csr(csr_content, _cryptography_backend)
        for sub in csr.subject:
            if sub.oid == cryptography.x509.oid.NameOID.COMMON_NAME:
                identifiers.add(('dns', sub.value))
        for extension in csr.extensions:
            if extension.oid == cryptography.x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME:
                for name in extension.value:
                    if isinstance(name, cryptography.x509.DNSName):
                        identifiers.add(('dns', name.value))
                    elif isinstance(name, cryptography.x509.IPAddress):
                        identifiers.add(('ip', name.value.compressed))
                    else:
                        raise BackendException('Found unsupported SAN identifier {0}'.format(name))
        return identifiers

    def get_cert_days(self, cert_filename=None, cert_content=None, now=None):
        """
        Return the days the certificate in cert_filename remains valid and -1
        if the file was not found. If cert_filename contains more than one
        certificate, only the first one will be considered.

        If now is not specified, datetime.datetime.now() is used.
        """
        if cert_filename is not None:
            cert_content = None
            if os.path.exists(cert_filename):
                cert_content = read_file(cert_filename)
        else:
            cert_content = to_bytes(cert_content)
        if cert_content is None:
            return -1
        cert_content = to_bytes(extract_first_pem(to_text(cert_content)) or '')
        try:
            cert = cryptography.x509.load_pem_x509_certificate(cert_content, _cryptography_backend)
        except Exception as e:
            if cert_filename is None:
                raise BackendException('Cannot parse certificate: {0}'.format(e))
            raise BackendException('Cannot parse certificate {0}: {1}'.format(cert_filename, e))
        if now is None:
            now = datetime.datetime.now()
        return (cert.not_valid_after - now).days

    def create_chain_matcher(self, criterium):
        """
        Given a Criterium object, creates a ChainMatcher object.
        """
        return CryptographyChainMatcher(criterium, self.module)