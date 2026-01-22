from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _Curve25519SHA256LibSSH:
    """
    As L{_Curve25519SHA256}, but with a pre-standardized algorithm name.
    """
    preference = 2
    hashProcessor = sha256