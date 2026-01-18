import warnings
from collections import namedtuple
def whathdr(filename):
    """Recognize sound headers."""
    with open(filename, 'rb') as f:
        h = f.read(512)
        for tf in tests:
            res = tf(h, f)
            if res:
                return SndHeaders(*res)
        return None