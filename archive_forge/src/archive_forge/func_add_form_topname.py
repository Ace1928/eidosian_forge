import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def add_form_topname(self, name: str) -> Optional[DictionaryObject]:
    """
        Add a top level form that groups all form fields below it.

        Args:
            name: text string of the "/T" Attribute of the created object

        Returns:
            The created object. ``None`` means no object was created.
        """
    catalog = self.root_object
    if '/AcroForm' not in catalog or not isinstance(catalog['/AcroForm'], DictionaryObject):
        return None
    acroform = cast(DictionaryObject, catalog[NameObject('/AcroForm')])
    if '/Fields' not in acroform:
        return None
    interim = DictionaryObject()
    interim[NameObject('/T')] = TextStringObject(name)
    interim[NameObject('/Kids')] = acroform[NameObject('/Fields')]
    self.cache_indirect_object(0, max([i for g, i in self.resolved_objects if g == 0]) + 1, interim)
    arr = ArrayObject()
    arr.append(interim.indirect_reference)
    acroform[NameObject('/Fields')] = arr
    for o in cast(ArrayObject, interim['/Kids']):
        obj = o.get_object()
        if '/Parent' in obj:
            logger_warning(f'Top Level Form Field {obj.indirect_reference} have a non-expected parent', __name__)
        obj[NameObject('/Parent')] = interim.indirect_reference
    return interim