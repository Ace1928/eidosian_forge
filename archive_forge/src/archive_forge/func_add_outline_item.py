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
def add_outline_item(self, title: str, page_number: Union[None, PageObject, IndirectObject, int], parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, color: Optional[Union[Tuple[float, float, float], str]]=None, bold: bool=False, italic: bool=False, fit: Fit=PAGE_FIT, is_open: bool=True) -> IndirectObject:
    """
        Add an outline item (commonly referred to as a "Bookmark") to the PDF file.

        Args:
            title: Title to use for this outline item.
            page_number: Page number this outline item will point to.
            parent: A reference to a parent outline item to create nested
                outline items.
            before:
            color: Color of the outline item's font as a red, green, blue tuple
                from 0.0 to 1.0 or as a Hex String (#RRGGBB)
            bold: Outline item font is bold
            italic: Outline item font is italic
            fit: The fit of the destination page.

        Returns:
            The added outline item as an indirect object.
        """
    page_ref: Union[None, NullObject, IndirectObject, NumberObject]
    if isinstance(italic, Fit):
        if fit is not None and page_number is None:
            page_number = fit
        return self.add_outline_item(title, page_number, parent, None, before, color, bold, italic, is_open=is_open)
    if page_number is None:
        action_ref = None
    else:
        if isinstance(page_number, IndirectObject):
            page_ref = page_number
        elif isinstance(page_number, PageObject):
            page_ref = page_number.indirect_reference
        elif isinstance(page_number, int):
            try:
                page_ref = self.pages[page_number].indirect_reference
            except IndexError:
                page_ref = NumberObject(page_number)
        if page_ref is None:
            logger_warning(f'can not find reference of page {page_number}', __name__)
            page_ref = NullObject()
        dest = Destination(NameObject('/' + title + ' outline item'), page_ref, fit)
        action_ref = self._add_object(DictionaryObject({NameObject(GoToActionArguments.D): dest.dest_array, NameObject(GoToActionArguments.S): NameObject('/GoTo')}))
    outline_item = self._add_object(_create_outline_item(action_ref, title, color, italic, bold))
    if parent is None:
        parent = self.get_outline_root()
    return self.add_outline_item_destination(outline_item, parent, before, is_open)