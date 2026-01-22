from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._encryption import Encryption
from ._page import PageObject
from ._reader import PdfReader
from ._utils import (
from ._writer import PdfWriter
from .constants import GoToActionArguments, TypArguments, TypFitArguments
from .constants import PagesAttributes as PA
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import LayoutType, OutlineType, PagemodeType
def _associate_outline_items_to_pages(self, pages: List[_MergedPage], outline: Optional[Iterable[OutlineItem]]=None) -> None:
    if outline is None:
        outline = self.outline
    assert outline is not None, 'hint for mypy'
    for outline_item in outline:
        if isinstance(outline_item, list):
            self._associate_outline_items_to_pages(pages, outline_item)
            continue
        page_index = None
        outline_item_page = outline_item['/Page']
        if isinstance(outline_item_page, NumberObject):
            continue
        for p in pages:
            if outline_item_page.get_object() == p.pagedata.get_object():
                page_index = p.id
        if page_index is not None:
            outline_item[NameObject('/Page')] = NumberObject(page_index)