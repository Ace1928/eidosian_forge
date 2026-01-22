from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def Bitwise(subcon):
    """converts the stream to bits, and passes the bitstream to subcon
    * subcon - a bitwise construct (usually BitField)
    """
    MAX_BUFFER = 1024 * 8

    def resizer(length):
        if length & 7:
            raise SizeofError('size must be a multiple of 8', length)
        return length >> 3
    if not subcon._is_flag(subcon.FLAG_DYNAMIC) and subcon.sizeof() < MAX_BUFFER:
        con = Buffered(subcon, encoder=decode_bin, decoder=encode_bin, resizer=resizer)
    else:
        con = Restream(subcon, stream_reader=BitStreamReader, stream_writer=BitStreamWriter, resizer=resizer)
    return con