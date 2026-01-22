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
def _write_dests(self) -> None:
    if self.output is None:
        raise RuntimeError(ERR_CLOSED_WRITER)
    for named_dest in self.named_dests:
        page_index = None
        if '/Page' in named_dest:
            for page_index, page in enumerate(self.pages):
                if page.id == named_dest['/Page']:
                    named_dest[NameObject('/Page')] = page.out_pagedata
                    break
        if page_index is not None:
            self.output.add_named_destination_object(named_dest)