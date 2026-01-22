from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
class _IEllipticCurveExchangeKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IEllipticCurveExchangeKexAlgorithm} describes a key exchange algorithm
    that uses an elliptic curve exchange between the client and server.
    """