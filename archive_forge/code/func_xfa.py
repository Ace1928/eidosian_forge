import struct
import zlib
from abc import abstractmethod
from datetime import datetime
from typing import (
from ._encryption import Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import (
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import (
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .errors import (
from .generic import (
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation
@property
def xfa(self) -> Optional[Dict[str, Any]]:
    tree: Optional[TreeObject] = None
    retval: Dict[str, Any] = {}
    catalog = self.root_object
    if '/AcroForm' not in catalog or not catalog['/AcroForm']:
        return None
    tree = cast(TreeObject, catalog['/AcroForm'])
    if '/XFA' in tree:
        fields = cast(ArrayObject, tree['/XFA'])
        i = iter(fields)
        for f in i:
            tag = f
            f = next(i)
            if isinstance(f, IndirectObject):
                field = cast(Optional[EncodedStreamObject], f.get_object())
                if field:
                    es = zlib.decompress(b_(field._data))
                    retval[tag] = es
    return retval