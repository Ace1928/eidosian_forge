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
def _get_filtered_outline(self, node: Any, pages: Dict[int, PageObject], reader: PdfReader) -> List[Destination]:
    """
        Extract outline item entries that are part of the specified page set.

        Args:
            node:
            pages:
            reader:

        Returns:
            A list of destination objects.
        """
    new_outline = []
    if node is None:
        node = NullObject()
    node = node.get_object()
    if node is None or isinstance(node, NullObject):
        node = DictionaryObject()
    if node.get('/Type', '') == '/Outlines' or '/Title' not in node:
        node = node.get('/First', None)
        if node is not None:
            node = node.get_object()
            new_outline += self._get_filtered_outline(node, pages, reader)
    else:
        v: Union[None, IndirectObject, NullObject]
        while node is not None:
            node = node.get_object()
            o = cast('Destination', reader._build_outline_item(node))
            v = self._get_cloned_page(cast('PageObject', o['/Page']), pages, reader)
            if v is None:
                v = NullObject()
            o[NameObject('/Page')] = v
            if '/First' in node:
                o._filtered_children = self._get_filtered_outline(node['/First'], pages, reader)
            else:
                o._filtered_children = []
            if not isinstance(o['/Page'], NullObject) or len(o._filtered_children) > 0:
                new_outline.append(o)
            node = node.get('/Next', None)
    return new_outline