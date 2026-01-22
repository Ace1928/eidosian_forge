from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def OnDemandPointer(offsetfunc, subcon, force_build=True):
    """an on-demand pointer.
    * offsetfunc - a function taking the context as an argument and returning
      the absolute stream position
    * subcon - the subcon that will be parsed from the `offsetfunc()` stream
      position on demand
    * force_build - see OnDemand. by default True.
    """
    return OnDemand(Pointer(offsetfunc, subcon), advance_stream=False, force_build=force_build)