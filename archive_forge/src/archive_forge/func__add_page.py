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
def _add_page(self, page: PageObject, action: Callable[[Any, Union[PageObject, IndirectObject]], None], excluded_keys: Iterable[str]=()) -> PageObject:
    assert cast(str, page[PA.TYPE]) == CO.PAGE
    page_org = page
    excluded_keys = list(excluded_keys)
    excluded_keys += [PA.PARENT, '/StructParents']
    try:
        del self._id_translated[id(page_org.indirect_reference.pdf)][page_org.indirect_reference.idnum]
    except Exception:
        pass
    page = cast('PageObject', page_org.clone(self, False, excluded_keys))
    if page_org.pdf is not None:
        other = page_org.pdf.pdf_header
        self.pdf_header = _get_max_pdf_version_header(self.pdf_header, other)
    page[NameObject(PA.PARENT)] = self._pages
    pages = cast(DictionaryObject, self.get_object(self._pages))
    assert page.indirect_reference is not None
    action(pages[PA.KIDS], page.indirect_reference)
    action(self.flattened_pages, page)
    page_count = cast(int, pages[PA.COUNT])
    pages[NameObject(PA.COUNT)] = NumberObject(page_count + 1)
    return page