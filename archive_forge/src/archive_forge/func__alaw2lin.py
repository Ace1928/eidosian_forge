import struct
import builtins
import warnings
from collections import namedtuple
def _alaw2lin(self, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import audioop
    return audioop.alaw2lin(data, 2)