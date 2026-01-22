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
def _trim_dests(self, pdf: PdfReader, dests: Dict[str, Dict[str, Any]], pages: Union[Tuple[int, int], Tuple[int, int, int], List[int]]) -> List[Dict[str, Any]]:
    """
        Remove named destinations that are not a part of the specified page set.

        Args:
            pdf:
            dests:
            pages:
        """
    new_dests = []
    lst = pages if isinstance(pages, list) else list(range(*pages))
    for key, obj in dests.items():
        for j in lst:
            if pdf.pages[j].get_object() == obj['/Page'].get_object():
                obj[NameObject('/Page')] = obj['/Page'].get_object()
                assert str_(key) == str_(obj['/Title'])
                new_dests.append(obj)
                break
    return new_dests