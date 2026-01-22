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
def _create_outline_item(action_ref: Union[None, IndirectObject], title: str, color: Union[Tuple[float, float, float], str, None], italic: bool, bold: bool) -> TreeObject:
    outline_item = TreeObject()
    if action_ref is not None:
        outline_item[NameObject('/A')] = action_ref
    outline_item.update({NameObject('/Title'): create_string_object(title)})
    if color:
        if isinstance(color, str):
            color = hex_to_rgb(color)
        outline_item.update({NameObject('/C'): ArrayObject([FloatObject(c) for c in color])})
    if italic or bold:
        format_flag = 0
        if italic:
            format_flag += 1
        if bold:
            format_flag += 2
        outline_item.update({NameObject('/F'): NumberObject(format_flag)})
    return outline_item