from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def _ru13(i):
    """reverse unsigned 13 bit number
    >>> print(_ru13(7936), _ru13(31), _ru13(47), _ru13(7808))
    31 7936 7808 47
    """
    r = 0
    for x in range(13):
        r <<= 1
        r |= i & 1
        i >>= 1
    return r