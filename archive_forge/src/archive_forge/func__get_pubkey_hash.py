import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse
import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from redis.exceptions import AuthorizationError, ConnectionError
def _get_pubkey_hash(certificate):
    pubkey = certificate.public_key()
    if isinstance(pubkey, RSAPublicKey):
        h = pubkey.public_bytes(Encoding.DER, PublicFormat.PKCS1)
    elif isinstance(pubkey, EllipticCurvePublicKey):
        h = pubkey.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    else:
        h = pubkey.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    sha1 = Hash(SHA1(), backend=backends.default_backend())
    sha1.update(h)
    return sha1.finalize()