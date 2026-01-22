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
def _get_cloned_page(self, page: Union[None, int, IndirectObject, PageObject, NullObject], pages: Dict[int, PageObject], reader: PdfReader) -> Optional[IndirectObject]:
    if isinstance(page, NullObject):
        return None
    if isinstance(page, int):
        _i = reader.pages[page].indirect_reference
    elif isinstance(page, DictionaryObject) and page.get('/Type', '') == '/Page':
        _i = page.indirect_reference
    elif isinstance(page, IndirectObject):
        _i = page
    try:
        return pages[_i.idnum].indirect_reference
    except Exception:
        return None