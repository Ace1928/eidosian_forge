import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def _write_xref_table(self, stream: StreamType, object_positions: List[int]) -> int:
    xref_location = stream.tell()
    stream.write(b'xref\n')
    stream.write(f'0 {len(self._objects) + 1}\n'.encode())
    stream.write(f'{0:0>10} {65535:0>5} f \n'.encode())
    for offset in object_positions:
        stream.write(f'{offset:0>10} {0:0>5} n \n'.encode())
    return xref_location