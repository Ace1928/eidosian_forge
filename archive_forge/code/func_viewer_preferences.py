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
def viewer_preferences(self) -> Optional[ViewerPreferences]:
    """Returns the existing ViewerPreferences as an overloaded dictionary."""
    o = self.root_object.get(CD.VIEWER_PREFERENCES, None)
    if o is None:
        return None
    o = o.get_object()
    if not isinstance(o, ViewerPreferences):
        o = ViewerPreferences(o)
        if hasattr(o, 'indirect_reference'):
            self._replace_object(o.indirect_reference, o)
        else:
            self.root_object[NameObject(CD.VIEWER_PREFERENCES)] = o
    return o