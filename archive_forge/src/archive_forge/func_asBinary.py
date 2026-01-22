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
def asBinary(self):
    """Get |ASN.1| value as a text string of bits.
        """
    binString = bin(self._value)[2:]
    return '0' * (len(self._value) - len(binString)) + binString