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
def _resolve_indirect_object(self, data: IndirectObject) -> IndirectObject:
    """
        Resolves an indirect object to an indirect object in this PDF file.

        If the input indirect object already belongs to this PDF file, it is
        returned directly. Otherwise, the object is retrieved from the input
        object's PDF file using the object's ID number and generation number. If
        the object cannot be found, a warning is logged and a `NullObject` is
        returned.

        If the object is not already in this PDF file, it is added to the file's
        list of objects and assigned a new ID number and generation number of 0.
        The hash value of the object is then added to the `_idnum_hash`
        dictionary, with the corresponding `IndirectObject` reference as the
        value.

        Args:
            data: The `IndirectObject` to resolve.

        Returns:
            The resolved `IndirectObject` in this PDF file.

        Raises:
            ValueError: If the input stream is closed.
        """
    if hasattr(data.pdf, 'stream') and data.pdf.stream.closed:
        raise ValueError(f'I/O operation on closed file: {data.pdf.stream.name}')
    if data.pdf == self:
        return data
    real_obj = data.pdf.get_object(data)
    if real_obj is None:
        logger_warning(f'Unable to resolve [{data.__class__.__name__}: {data}], returning NullObject instead', __name__)
        real_obj = NullObject()
    hash_value = real_obj.hash_value()
    if hash_value in self._idnum_hash:
        return self._idnum_hash[hash_value]
    if data.pdf == self:
        self._idnum_hash[hash_value] = IndirectObject(data.idnum, 0, self)
    else:
        self._idnum_hash[hash_value] = self._add_object(real_obj)
    return self._idnum_hash[hash_value]