from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def _crc11(value):
    """
    >>> usps = [USPS_4State('01234567094987654321',x).binary for x in ('','01234','012345678','01234567891')]
    >>> print(' '.join(nhex(x) for x in usps))
    0x1122103b5c2004b1 0xd138a87bab5cf3804b1 0x202bdc097711204d21804b1 0x16907b2a24abc16a2e5c004b1
    >>> print(' '.join(nhex(_crc11(x)) for x in usps))
    0x51 0x65 0x606 0x751
    """
    hexbytes = nhex(int(value))[2:]
    hexbytes = '0' * (26 - len(hexbytes)) + hexbytes
    gp = 3893
    fcs = 2047
    data = int(hexbytes[:2], 16) << 5
    for b in range(2, 8):
        if (fcs ^ data) & 1024:
            fcs = fcs << 1 ^ gp
        else:
            fcs = fcs << 1
        fcs &= 2047
        data <<= 1
    for x in range(2, 2 * 13, 2):
        data = int(hexbytes[x:x + 2], 16) << 3
        for b in range(8):
            if (fcs ^ data) & 1024:
                fcs = fcs << 1 ^ gp
            else:
                fcs = fcs << 1
            fcs &= 2047
            data <<= 1
    return fcs