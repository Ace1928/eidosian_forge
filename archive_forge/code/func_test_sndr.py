import warnings
from collections import namedtuple
def test_sndr(h, f):
    """SNDR file"""
    if h.startswith(b'\x00\x00'):
        rate = get_short_le(h[2:4])
        if 4000 <= rate <= 25000:
            return ('sndr', rate, 1, -1, 8)