import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def _get_ids_image(self, obj: Optional[DictionaryObject]=None, ancest: Optional[List[str]]=None, call_stack: Optional[List[Any]]=None) -> List[Union[str, List[str]]]:
    if call_stack is None:
        call_stack = []
    _i = getattr(obj, 'indirect_reference', None)
    if _i in call_stack:
        return []
    else:
        call_stack.append(_i)
    if self.inline_images_keys is None:
        content = self._get_contents_as_bytes() or b''
        nb_inlines = 0
        for matching in re.finditer(WHITESPACES_AS_REGEXP + b'BI' + WHITESPACES_AS_REGEXP, content):
            start_of_string = content[:matching.start()]
            if len(re.findall(b'[^\\\\]\\(', start_of_string)) == len(re.findall(b'[^\\\\]\\)', start_of_string)):
                nb_inlines += 1
        self.inline_images_keys = [f'~{x}~' for x in range(nb_inlines)]
    if obj is None:
        obj = self
    if ancest is None:
        ancest = []
    lst: List[Union[str, List[str]]] = []
    if PG.RESOURCES not in obj or RES.XOBJECT not in cast(DictionaryObject, obj[PG.RESOURCES]):
        return self.inline_images_keys
    x_object = obj[PG.RESOURCES][RES.XOBJECT].get_object()
    for o in x_object:
        if not isinstance(x_object[o], StreamObject):
            continue
        if x_object[o][IA.SUBTYPE] == '/Image':
            lst.append(o if len(ancest) == 0 else ancest + [o])
        else:
            lst.extend(self._get_ids_image(x_object[o], ancest + [o], call_stack))
    return lst + self.inline_images_keys