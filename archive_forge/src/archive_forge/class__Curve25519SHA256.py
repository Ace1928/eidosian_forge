from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _Curve25519SHA256:
    """
    Elliptic Curve Key Exchange using Curve25519 and SHA256. Defined in
    U{https://datatracker.ietf.org/doc/draft-ietf-curdle-ssh-curves/}.
    """
    preference = 1
    hashProcessor = sha256