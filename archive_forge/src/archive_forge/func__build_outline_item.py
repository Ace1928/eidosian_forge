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
def _build_outline_item(self, node: DictionaryObject) -> Optional[Destination]:
    dest, title, outline_item = (None, None, None)
    try:
        title = cast('str', node['/Title'])
    except KeyError:
        if self.strict:
            raise PdfReadError(f'Outline Entry Missing /Title attribute: {node!r}')
        title = ''
    if '/A' in node:
        action = cast(DictionaryObject, node['/A'])
        action_type = cast(NameObject, action[GoToActionArguments.S])
        if action_type == '/GoTo':
            dest = action[GoToActionArguments.D]
    elif '/Dest' in node:
        dest = node['/Dest']
        if isinstance(dest, DictionaryObject) and '/D' in dest:
            dest = dest['/D']
    if isinstance(dest, ArrayObject):
        outline_item = self._build_destination(title, dest)
    elif isinstance(dest, str):
        try:
            outline_item = self._build_destination(title, self._namedDests[dest].dest_array)
        except KeyError:
            outline_item = self._build_destination(title, None)
    elif dest is None:
        outline_item = self._build_destination(title, dest)
    else:
        if self.strict:
            raise PdfReadError(f'Unexpected destination {dest!r}')
        else:
            logger_warning(f'Removed unexpected destination {dest!r} from destination', __name__)
        outline_item = self._build_destination(title, None)
    if outline_item:
        if '/C' in node:
            outline_item[NameObject('/C')] = ArrayObject((FloatObject(c) for c in node['/C']))
        if '/F' in node:
            outline_item[NameObject('/F')] = node['/F']
        if '/Count' in node:
            outline_item[NameObject('/Count')] = node['/Count']
        outline_item[NameObject('/%is_open%')] = BooleanObject(node.get('/Count', 0) >= 0)
    outline_item.node = node
    try:
        outline_item.indirect_reference = node.indirect_reference
    except AttributeError:
        pass
    return outline_item