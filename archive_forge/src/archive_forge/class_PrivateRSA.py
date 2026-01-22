import math
import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
class PrivateRSA(CryptographyPrivateKey):
    key: rsa.RSAPrivateKey
    key_cls = rsa.RSAPrivateKey
    public_cls = PublicRSA
    default_public_exponent = 65537

    def sign(self, data: bytes, verify: bool=False) -> bytes:
        """Sign using a private key per RFC 3110, section 3."""
        signature = self.key.sign(data, padding.PKCS1v15(), self.public_cls.chosen_hash)
        if verify:
            self.public_key().verify(signature, data)
        return signature

    @classmethod
    def generate(cls, key_size: int) -> 'PrivateRSA':
        return cls(key=rsa.generate_private_key(public_exponent=cls.default_public_exponent, key_size=key_size, backend=default_backend()))