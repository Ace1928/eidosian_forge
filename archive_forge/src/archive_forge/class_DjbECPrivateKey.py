import struct
from .ec import ECPublicKey, ECPrivateKey
from ..util.byteutil import ByteUtil
class DjbECPrivateKey(ECPrivateKey):

    def __init__(self, privateKey):
        self.privateKey = privateKey

    def getType(self):
        from .curve import Curve
        return Curve.DJB_TYPE

    def getPrivateKey(self):
        return self.privateKey

    def serialize(self):
        return self.privateKey

    def __eq__(self, other):
        return self.privateKey == other.getPrivateKey()