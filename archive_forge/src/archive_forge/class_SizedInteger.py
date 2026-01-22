import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class SizedInteger(SizedIntegerBase):
    bitLength = leadingZeroBits = None

    def setBitLength(self, bitLength):
        self.bitLength = bitLength
        self.leadingZeroBits = max(bitLength - integer.bitLength(self), 0)
        return self

    def __len__(self):
        if self.bitLength is None:
            self.setBitLength(integer.bitLength(self))
        return self.bitLength