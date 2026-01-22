import logging
import struct
import sys
from io import BytesIO
from typing import (
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .cmapdb import CMapParser
from .cmapdb import FileUnicodeMap
from .cmapdb import IdentityUnicodeMap
from .cmapdb import UnicodeMap
from .encodingdb import EncodingDB
from .encodingdb import name2unicode
from .fontmetrics import FONT_METRICS
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import num_value
from .pdftypes import resolve1, resolve_all
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import literal_name
from .utils import Matrix, Point
from .utils import Rect
from .utils import apply_matrix_norm
from .utils import choplist
from .utils import nunpack
@staticmethod
def _get_cmap_name(spec: Mapping[str, Any], strict: bool) -> str:
    """Get cmap name from font specification"""
    cmap_name = 'unknown'
    try:
        spec_encoding = spec['Encoding']
        if hasattr(spec_encoding, 'name'):
            cmap_name = literal_name(spec['Encoding'])
        else:
            cmap_name = literal_name(spec_encoding['CMapName'])
    except KeyError:
        if strict:
            raise PDFFontError('Encoding is unspecified')
    if type(cmap_name) is PDFStream:
        cmap_name_stream: PDFStream = cast(PDFStream, cmap_name)
        if 'CMapName' in cmap_name_stream:
            cmap_name = cmap_name_stream.get('CMapName').name
        elif strict:
            raise PDFFontError('CMapName unspecified for encoding')
    return IDENTITY_ENCODER.get(cmap_name, cmap_name)