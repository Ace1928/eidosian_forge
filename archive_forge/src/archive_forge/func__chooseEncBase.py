import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
def _chooseEncBase(self, value):
    m, b, e = value
    encBase = [2, 8, 16]
    if value.binEncBase in encBase:
        return self._dropFloatingPoint(m, value.binEncBase, e)
    elif self.binEncBase in encBase:
        return self._dropFloatingPoint(m, self.binEncBase, e)
    mantissa = [m, m, m]
    exponent = [e, e, e]
    sign = 1
    encbase = 2
    e = float('inf')
    for i in range(3):
        sign, mantissa[i], encBase[i], exponent[i] = self._dropFloatingPoint(mantissa[i], encBase[i], exponent[i])
        if abs(exponent[i]) < abs(e) or (abs(exponent[i]) == abs(e) and mantissa[i] < m):
            e = exponent[i]
            m = int(mantissa[i])
            encbase = encBase[i]
    if LOG:
        LOG('automatically chosen REAL encoding base %s, sign %s, mantissa %s, exponent %s' % (encbase, sign, m, e))
    return (sign, m, encbase, e)