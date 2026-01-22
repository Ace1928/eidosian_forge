from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def _initNof13Table(N, lenT):
    """create and return table of 13 bit values with N bits on
    >>> T = _initNof13Table(5,1287)
    >>> print(' '.join('T[%d]=%d' % (i, T[i]) for i in (0,1,2,3,4,1271,1272,1284,1285,1286)))
    T[0]=31 T[1]=7936 T[2]=47 T[3]=7808 T[4]=55 T[1271]=6275 T[1272]=6211 T[1284]=856 T[1285]=744 T[1286]=496
    """
    T = lenT * [None]
    l = 0
    u = lenT - 1
    for c in range(8192):
        bc = 0
        for b in range(13):
            bc += c & 1 << b != 0
        if bc != N:
            continue
        r = _ru13(c)
        if r < c:
            continue
        if r == c:
            T[u] = c
            u -= 1
        else:
            T[l] = c
            l += 1
            T[l] = r
            l += 1
    assert l == u + 1, 'u+1(%d)!=l(%d) for %d of 13 table' % (u + 1, l, N)
    return T