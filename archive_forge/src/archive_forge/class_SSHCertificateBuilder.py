from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
class SSHCertificateBuilder:

    def __init__(self, _public_key: typing.Optional[SSHCertPublicKeyTypes]=None, _serial: typing.Optional[int]=None, _type: typing.Optional[SSHCertificateType]=None, _key_id: typing.Optional[bytes]=None, _valid_principals: typing.List[bytes]=[], _valid_for_all_principals: bool=False, _valid_before: typing.Optional[int]=None, _valid_after: typing.Optional[int]=None, _critical_options: typing.List[typing.Tuple[bytes, bytes]]=[], _extensions: typing.List[typing.Tuple[bytes, bytes]]=[]):
        self._public_key = _public_key
        self._serial = _serial
        self._type = _type
        self._key_id = _key_id
        self._valid_principals = _valid_principals
        self._valid_for_all_principals = _valid_for_all_principals
        self._valid_before = _valid_before
        self._valid_after = _valid_after
        self._critical_options = _critical_options
        self._extensions = _extensions

    def public_key(self, public_key: SSHCertPublicKeyTypes) -> SSHCertificateBuilder:
        if not isinstance(public_key, (ec.EllipticCurvePublicKey, rsa.RSAPublicKey, ed25519.Ed25519PublicKey)):
            raise TypeError('Unsupported key type')
        if self._public_key is not None:
            raise ValueError('public_key already set')
        return SSHCertificateBuilder(_public_key=public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def serial(self, serial: int) -> SSHCertificateBuilder:
        if not isinstance(serial, int):
            raise TypeError('serial must be an integer')
        if not 0 <= serial < 2 ** 64:
            raise ValueError('serial must be between 0 and 2**64')
        if self._serial is not None:
            raise ValueError('serial already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def type(self, type: SSHCertificateType) -> SSHCertificateBuilder:
        if not isinstance(type, SSHCertificateType):
            raise TypeError('type must be an SSHCertificateType')
        if self._type is not None:
            raise ValueError('type already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def key_id(self, key_id: bytes) -> SSHCertificateBuilder:
        if not isinstance(key_id, bytes):
            raise TypeError('key_id must be bytes')
        if self._key_id is not None:
            raise ValueError('key_id already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def valid_principals(self, valid_principals: typing.List[bytes]) -> SSHCertificateBuilder:
        if self._valid_for_all_principals:
            raise ValueError("Principals can't be set because the cert is valid for all principals")
        if not all((isinstance(x, bytes) for x in valid_principals)) or not valid_principals:
            raise TypeError("principals must be a list of bytes and can't be empty")
        if self._valid_principals:
            raise ValueError('valid_principals already set')
        if len(valid_principals) > _SSHKEY_CERT_MAX_PRINCIPALS:
            raise ValueError('Reached or exceeded the maximum number of valid_principals')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def valid_for_all_principals(self):
        if self._valid_principals:
            raise ValueError("valid_principals already set, can't set valid_for_all_principals")
        if self._valid_for_all_principals:
            raise ValueError('valid_for_all_principals already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=True, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def valid_before(self, valid_before: typing.Union[int, float]) -> SSHCertificateBuilder:
        if not isinstance(valid_before, (int, float)):
            raise TypeError('valid_before must be an int or float')
        valid_before = int(valid_before)
        if valid_before < 0 or valid_before >= 2 ** 64:
            raise ValueError('valid_before must [0, 2**64)')
        if self._valid_before is not None:
            raise ValueError('valid_before already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def valid_after(self, valid_after: typing.Union[int, float]) -> SSHCertificateBuilder:
        if not isinstance(valid_after, (int, float)):
            raise TypeError('valid_after must be an int or float')
        valid_after = int(valid_after)
        if valid_after < 0 or valid_after >= 2 ** 64:
            raise ValueError('valid_after must [0, 2**64)')
        if self._valid_after is not None:
            raise ValueError('valid_after already set')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=valid_after, _critical_options=self._critical_options, _extensions=self._extensions)

    def add_critical_option(self, name: bytes, value: bytes) -> SSHCertificateBuilder:
        if not isinstance(name, bytes) or not isinstance(value, bytes):
            raise TypeError('name and value must be bytes')
        if name in [name for name, _ in self._critical_options]:
            raise ValueError('Duplicate critical option name')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options + [(name, value)], _extensions=self._extensions)

    def add_extension(self, name: bytes, value: bytes) -> SSHCertificateBuilder:
        if not isinstance(name, bytes) or not isinstance(value, bytes):
            raise TypeError('name and value must be bytes')
        if name in [name for name, _ in self._extensions]:
            raise ValueError('Duplicate extension name')
        return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions + [(name, value)])

    def sign(self, private_key: SSHCertPrivateKeyTypes) -> SSHCertificate:
        if not isinstance(private_key, (ec.EllipticCurvePrivateKey, rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
            raise TypeError('Unsupported private key type')
        if self._public_key is None:
            raise ValueError('public_key must be set')
        serial = 0 if self._serial is None else self._serial
        if self._type is None:
            raise ValueError('type must be set')
        key_id = b'' if self._key_id is None else self._key_id
        if not self._valid_principals and (not self._valid_for_all_principals):
            raise ValueError('valid_principals must be set if valid_for_all_principals is False')
        if self._valid_before is None:
            raise ValueError('valid_before must be set')
        if self._valid_after is None:
            raise ValueError('valid_after must be set')
        if self._valid_after > self._valid_before:
            raise ValueError('valid_after must be earlier than valid_before')
        self._critical_options.sort(key=lambda x: x[0])
        self._extensions.sort(key=lambda x: x[0])
        key_type = _get_ssh_key_type(self._public_key)
        cert_prefix = key_type + _CERT_SUFFIX
        nonce = os.urandom(32)
        kformat = _lookup_kformat(key_type)
        f = _FragList()
        f.put_sshstr(cert_prefix)
        f.put_sshstr(nonce)
        kformat.encode_public(self._public_key, f)
        f.put_u64(serial)
        f.put_u32(self._type.value)
        f.put_sshstr(key_id)
        fprincipals = _FragList()
        for p in self._valid_principals:
            fprincipals.put_sshstr(p)
        f.put_sshstr(fprincipals.tobytes())
        f.put_u64(self._valid_after)
        f.put_u64(self._valid_before)
        fcrit = _FragList()
        for name, value in self._critical_options:
            fcrit.put_sshstr(name)
            if len(value) > 0:
                foptval = _FragList()
                foptval.put_sshstr(value)
                fcrit.put_sshstr(foptval.tobytes())
            else:
                fcrit.put_sshstr(value)
        f.put_sshstr(fcrit.tobytes())
        fext = _FragList()
        for name, value in self._extensions:
            fext.put_sshstr(name)
            if len(value) > 0:
                fextval = _FragList()
                fextval.put_sshstr(value)
                fext.put_sshstr(fextval.tobytes())
            else:
                fext.put_sshstr(value)
        f.put_sshstr(fext.tobytes())
        f.put_sshstr(b'')
        ca_type = _get_ssh_key_type(private_key)
        caformat = _lookup_kformat(ca_type)
        caf = _FragList()
        caf.put_sshstr(ca_type)
        caformat.encode_public(private_key.public_key(), caf)
        f.put_sshstr(caf.tobytes())
        if isinstance(private_key, ed25519.Ed25519PrivateKey):
            signature = private_key.sign(f.tobytes())
            fsig = _FragList()
            fsig.put_sshstr(ca_type)
            fsig.put_sshstr(signature)
            f.put_sshstr(fsig.tobytes())
        elif isinstance(private_key, ec.EllipticCurvePrivateKey):
            hash_alg = _get_ec_hash_alg(private_key.curve)
            signature = private_key.sign(f.tobytes(), ec.ECDSA(hash_alg))
            r, s = asym_utils.decode_dss_signature(signature)
            fsig = _FragList()
            fsig.put_sshstr(ca_type)
            fsigblob = _FragList()
            fsigblob.put_mpint(r)
            fsigblob.put_mpint(s)
            fsig.put_sshstr(fsigblob.tobytes())
            f.put_sshstr(fsig.tobytes())
        else:
            assert isinstance(private_key, rsa.RSAPrivateKey)
            fsig = _FragList()
            fsig.put_sshstr(_SSH_RSA_SHA512)
            signature = private_key.sign(f.tobytes(), padding.PKCS1v15(), hashes.SHA512())
            fsig.put_sshstr(signature)
            f.put_sshstr(fsig.tobytes())
        cert_data = binascii.b2a_base64(f.tobytes()).strip()
        return typing.cast(SSHCertificate, load_ssh_public_identity(b''.join([cert_prefix, b' ', cert_data])))