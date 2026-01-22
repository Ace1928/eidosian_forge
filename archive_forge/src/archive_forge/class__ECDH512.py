from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _ECDH512:
    """
    Elliptic Curve Key Exchange with SHA-512 as HASH. Defined in
    RFC 5656.
    """
    preference = 5
    hashProcessor = sha512