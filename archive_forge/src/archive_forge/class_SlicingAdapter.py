from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class SlicingAdapter(Adapter):
    """
    Adapter for slicing a list (getting a slice from that list)

    Parameters:
    * subcon - the subcon to slice
    * start - start index
    * stop - stop index (or None for up-to-end)
    * step - step (or None for every element)
    """
    __slots__ = ['start', 'stop', 'step']

    def __init__(self, subcon, start, stop=None):
        Adapter.__init__(self, subcon)
        self.start = start
        self.stop = stop

    def _encode(self, obj, context):
        if self.start is None:
            return obj
        return [None] * self.start + obj

    def _decode(self, obj, context):
        return obj[self.start:self.stop]