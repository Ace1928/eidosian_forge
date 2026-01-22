from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class HexDumpAdapter(Adapter):
    """
    Adapter for hex-dumping strings. It returns a HexString, which is a string
    """
    __slots__ = ['linesize']

    def __init__(self, subcon, linesize=16):
        Adapter.__init__(self, subcon)
        self.linesize = linesize

    def _encode(self, obj, context):
        return obj

    def _decode(self, obj, context):
        return HexString(obj, linesize=self.linesize)