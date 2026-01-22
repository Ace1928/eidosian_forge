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
def __setitem__(self, idx, value):
    if octets.isStringType(idx):
        try:
            self.setComponentByName(idx, value)
        except error.PyAsn1Error:
            raise KeyError(sys.exc_info()[1])
    else:
        try:
            self.setComponentByPosition(idx, value)
        except error.PyAsn1Error:
            raise IndexError(sys.exc_info()[1])