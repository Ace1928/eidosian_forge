import math
import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ._utils import (
from .constants import CcittFaxDecodeParameters as CCITT
from .constants import ColorSpaces
from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import LzwFilterParameters as LZW
from .constants import StreamAttributes as SA
from .errors import DeprecationError, PdfReadError, PdfStreamError
from .generic import (
class RunLengthDecode:
    """
    The RunLengthDecode filter decodes data that has been encoded in a
    simple byte-oriented format based on run length.
    The encoded data is a sequence of runs, where each run consists of
    a length byte followed by 1 to 128 bytes of data. If the length byte is
    in the range 0 to 127,
    the following length + 1 (1 to 128) bytes are copied literally during
    decompression.
    If length is in the range 129 to 255, the following single byte is to be
    copied 257 − length (2 to 128) times during decompression. A length value
    of 128 denotes EOD.
    """

    @staticmethod
    def decode(data: bytes, decode_parms: Optional[DictionaryObject]=None, **kwargs: Any) -> bytes:
        """
        Decode a run length encoded data stream.

        Args:
          data: a bytes sequence of length/data
          decode_parms: ignored.

        Returns:
          A bytes decompressed sequence.

        Raises:
          PdfStreamError:
        """
        lst = []
        index = 0
        while True:
            if index >= len(data):
                logger_warning('missing EOD in RunLengthDecode, check if output is OK', __name__)
                break
            length = data[index]
            index += 1
            if length == 128:
                if index < len(data):
                    raise PdfStreamError('early EOD in RunLengthDecode')
                else:
                    break
            elif length < 128:
                length += 1
                lst.append(data[index:index + length])
                index += length
            else:
                length = 257 - length
                lst.append(bytes((data[index],)) * length)
                index += 1
        return b''.join(lst)