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
def _sweep_indirect_references(self, root: Union[ArrayObject, BooleanObject, DictionaryObject, FloatObject, IndirectObject, NameObject, PdfObject, NumberObject, TextStringObject, NullObject]) -> None:
    """
        Resolving any circular references to Page objects.

        Circular references to Page objects can arise when objects such as
        annotations refer to their associated page. If these references are not
        properly handled, the PDF file will contain multiple copies of the same
        Page object. To address this problem, Page objects store their original
        object reference number. This method adds the reference number of any
        circularly referenced Page objects to an external reference map. This
        ensures that self-referencing trees reference the correct new object
        location, rather than copying in a new copy of the Page object.

        Args:
            root: The root of the PDF object tree to sweep.
        """
    stack: Deque[Tuple[Any, Optional[Any], Any, List[PdfObject]]] = collections.deque()
    discovered = []
    parent = None
    grant_parents: List[PdfObject] = []
    key_or_id = None
    stack.append((root, parent, key_or_id, grant_parents))
    while len(stack):
        data, parent, key_or_id, grant_parents = stack.pop()
        if isinstance(data, (ArrayObject, DictionaryObject)):
            for key, value in data.items():
                stack.append((value, data, key, grant_parents + [parent] if parent is not None else []))
        elif isinstance(data, IndirectObject) and data.pdf != self:
            data = self._resolve_indirect_object(data)
            if str(data) not in discovered:
                discovered.append(str(data))
                stack.append((data.get_object(), None, None, []))
        if isinstance(parent, (DictionaryObject, ArrayObject)):
            if isinstance(data, StreamObject):
                data = self._resolve_indirect_object(self._add_object(data))
            update_hashes = []
            if parent[key_or_id] != data:
                update_hashes = [parent.hash_value()] + [grant_parent.hash_value() for grant_parent in grant_parents]
                parent[key_or_id] = data
            for old_hash in update_hashes:
                indirect_reference = self._idnum_hash.pop(old_hash, None)
                if indirect_reference is not None:
                    indirect_reference_obj = indirect_reference.get_object()
                    if indirect_reference_obj is not None:
                        self._idnum_hash[indirect_reference_obj.hash_value()] = indirect_reference