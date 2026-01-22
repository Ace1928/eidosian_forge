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
def add_outline_item_destination(self, page_destination: Union[IndirectObject, PageObject, TreeObject], parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, is_open: bool=True) -> IndirectObject:
    page_destination = cast(PageObject, page_destination.get_object())
    if isinstance(page_destination, PageObject):
        return self.add_outline_item_destination(Destination(f'page #{page_destination.page_number}', cast(IndirectObject, page_destination.indirect_reference), Fit.fit()))
    if parent is None:
        parent = self.get_outline_root()
    page_destination[NameObject('/%is_open%')] = BooleanObject(is_open)
    parent = cast(TreeObject, parent.get_object())
    page_destination_ref = self._add_object(page_destination)
    if before is not None:
        before = before.indirect_reference
    parent.insert_child(page_destination_ref, before, self, page_destination.inc_parent_counter_outline if is_open else lambda x, y: 0)
    if '/Count' not in page_destination:
        page_destination[NameObject('/Count')] = NumberObject(0)
    return page_destination_ref