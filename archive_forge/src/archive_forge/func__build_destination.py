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
def _build_destination(self, title: str, array: Optional[List[Union[NumberObject, IndirectObject, None, NullObject, DictionaryObject]]]) -> Destination:
    page, typ = (None, None)
    if isinstance(array, (NullObject, str)) or (isinstance(array, ArrayObject) and len(array) == 0) or array is None:
        page = NullObject()
        return Destination(title, page, Fit.fit())
    else:
        page, typ = array[0:2]
        array = array[2:]
        try:
            return Destination(title, page, Fit(fit_type=typ, fit_args=array))
        except PdfReadError:
            logger_warning(f'Unknown destination: {title} {array}', __name__)
            if self.strict:
                raise
            tmp = self.pages[0].indirect_reference
            indirect_reference = NullObject() if tmp is None else tmp
            return Destination(title, indirect_reference, Fit.fit())