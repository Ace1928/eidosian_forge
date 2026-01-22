import struct
import builtins
import warnings
from collections import namedtuple
def _lin2ulaw(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    return audioop.lin2ulaw(data, 2)