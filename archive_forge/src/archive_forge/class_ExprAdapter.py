from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class ExprAdapter(Adapter):
    """
    A generic adapter that accepts 'encoder' and 'decoder' as parameters. You
    can use ExprAdapter instead of writing a full-blown class when only a
    simple expression is needed.

    Parameters:
    * subcon - the subcon to adapt
    * encoder - a function that takes (obj, context) and returns an encoded
      version of obj
    * decoder - a function that takes (obj, context) and returns a decoded
      version of obj

    Example:
    ExprAdapter(UBInt8("foo"),
        encoder = lambda obj, ctx: obj / 4,
        decoder = lambda obj, ctx: obj * 4,
    )
    """
    __slots__ = ['_encode', '_decode']

    def __init__(self, subcon, encoder, decoder):
        Adapter.__init__(self, subcon)
        self._encode = encoder
        self._decode = decoder