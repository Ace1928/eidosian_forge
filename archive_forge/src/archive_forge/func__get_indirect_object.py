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
def _get_indirect_object(self, num: int, gen: int) -> Optional[PdfObject]:
    """
        Used to ease development.

        This is equivalent to generic.IndirectObject(num,gen,self).get_object()

        Args:
            num: The object number of the indirect object.
            gen: The generation number of the indirect object.

        Returns:
            A PdfObject
        """
    return IndirectObject(num, gen, self).get_object()