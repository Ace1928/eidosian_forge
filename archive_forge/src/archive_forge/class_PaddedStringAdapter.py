from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class PaddedStringAdapter(Adapter):
    """
    Adapter for padded strings.
    See String.

    Parameters:
    * subcon - the subcon to adapt
    * padchar - the padding character. default is b"\\x00".
    * paddir - the direction where padding is placed ("right", "left", or
      "center"). the default is "right".
    * trimdir - the direction where trimming will take place ("right" or
      "left"). the default is "right". trimming is only meaningful for
      building, when the given string is too long.
    """
    __slots__ = ['padchar', 'paddir', 'trimdir']

    def __init__(self, subcon, padchar=b'\x00', paddir='right', trimdir='right'):
        if paddir not in ('right', 'left', 'center'):
            raise ValueError("paddir must be 'right', 'left' or 'center'", paddir)
        if trimdir not in ('right', 'left'):
            raise ValueError("trimdir must be 'right' or 'left'", trimdir)
        Adapter.__init__(self, subcon)
        self.padchar = padchar
        self.paddir = paddir
        self.trimdir = trimdir

    def _decode(self, obj, context):
        if self.paddir == 'right':
            obj = obj.rstrip(self.padchar)
        elif self.paddir == 'left':
            obj = obj.lstrip(self.padchar)
        else:
            obj = obj.strip(self.padchar)
        return obj

    def _encode(self, obj, context):
        size = self._sizeof(context)
        if self.paddir == 'right':
            obj = obj.ljust(size, self.padchar)
        elif self.paddir == 'left':
            obj = obj.rjust(size, self.padchar)
        else:
            obj = obj.center(size, self.padchar)
        if len(obj) > size:
            if self.trimdir == 'right':
                obj = obj[:size]
            else:
                obj = obj[-size:]
        return obj