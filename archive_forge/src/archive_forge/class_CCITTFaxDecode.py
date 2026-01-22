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
class CCITTFaxDecode:
    """
    See 3.3.5 CCITTFaxDecode Filter (PDF 1.7 Standard).

    Either Group 3 or Group 4 CCITT facsimile (fax) encoding.
    CCITT encoding is bit-oriented, not byte-oriented.

    See: TABLE 3.9 Optional parameters for the CCITTFaxDecode filter
    """

    @staticmethod
    def _get_parameters(parameters: Union[None, ArrayObject, DictionaryObject, IndirectObject], rows: int) -> CCITParameters:
        k = 0
        columns = 1728
        if parameters:
            parameters_unwrapped = cast(Union[ArrayObject, DictionaryObject], parameters.get_object())
            if isinstance(parameters_unwrapped, ArrayObject):
                for decode_parm in parameters_unwrapped:
                    if CCITT.COLUMNS in decode_parm:
                        columns = decode_parm[CCITT.COLUMNS]
                    if CCITT.K in decode_parm:
                        k = decode_parm[CCITT.K]
            else:
                if CCITT.COLUMNS in parameters_unwrapped:
                    columns = parameters_unwrapped[CCITT.COLUMNS]
                if CCITT.K in parameters_unwrapped:
                    k = parameters_unwrapped[CCITT.K]
        return CCITParameters(k, columns, rows)

    @staticmethod
    def decode(data: bytes, decode_parms: Optional[DictionaryObject]=None, height: int=0, **kwargs: Any) -> bytes:
        if 'decodeParms' in kwargs:
            deprecate_with_replacement('decodeParms', 'parameters', '4.0.0')
            decode_parms = kwargs['decodeParms']
        if isinstance(decode_parms, ArrayObject):
            deprecation_no_replacement('decode_parms being an ArrayObject', removed_in='3.15.5')
        params = CCITTFaxDecode._get_parameters(decode_parms, height)
        img_size = len(data)
        tiff_header_struct = '<2shlh' + 'hhll' * 8 + 'h'
        tiff_header = struct.pack(tiff_header_struct, b'II', 42, 8, 8, 256, 4, 1, params.columns, 257, 4, 1, params.rows, 258, 3, 1, 1, 259, 3, 1, params.group, 262, 3, 1, 0, 273, 4, 1, struct.calcsize(tiff_header_struct), 278, 4, 1, params.rows, 279, 4, 1, img_size, 0)
        return tiff_header + data