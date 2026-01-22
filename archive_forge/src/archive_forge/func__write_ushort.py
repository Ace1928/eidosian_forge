import struct
import builtins
import warnings
from collections import namedtuple
def _write_ushort(f, x):
    f.write(struct.pack('>H', x))