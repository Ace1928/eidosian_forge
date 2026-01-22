from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class CStringAdapter(StringAdapter):
    """
    Adapter for C-style strings (strings terminated by a terminator char).

    Parameters:
    * subcon - the subcon to convert
    * terminators - a sequence of terminator chars. default is b"\\x00".
    * encoding - the character encoding to use (e.g., "utf8"), or None to
      return raw-bytes. the terminator characters are not affected by the
      encoding.
    """
    __slots__ = ['terminators']

    def __init__(self, subcon, terminators=b'\x00', encoding=None):
        StringAdapter.__init__(self, subcon, encoding=encoding)
        self.terminators = terminators

    def _encode(self, obj, context):
        return StringAdapter._encode(self, obj, context) + self.terminators[0:1]

    def _decode(self, obj, context):
        return StringAdapter._decode(self, b''.join(obj[:-1]), context)