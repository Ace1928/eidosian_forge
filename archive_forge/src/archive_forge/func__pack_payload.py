import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
def _pack_payload(data: typing.Any, b_payload: bytearray, payload_offset: int, pack_func: typing.Optional[typing.Callable[[typing.Any], bytes]]=None) -> typing.Tuple[bytes, int]:
    if data:
        b_data = pack_func(data) if pack_func else data
    else:
        b_data = b''
    b_payload.extend(b_data)
    length = len(b_data)
    b_field = struct.pack('<H', length) * 2 + struct.pack('<I', payload_offset)
    payload_offset += length
    return (b_field, payload_offset)