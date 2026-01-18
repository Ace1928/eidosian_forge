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
def to_unichr(self, cid: int) -> str:
    try:
        if not self.unicode_map:
            raise KeyError(cid)
        return self.unicode_map.get_unichr(cid)
    except KeyError:
        raise PDFUnicodeNotDefined(self.cidcoding, cid)