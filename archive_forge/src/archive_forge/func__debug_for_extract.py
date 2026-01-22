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
def _debug_for_extract(self) -> str:
    out = ''
    for ope, op in ContentStream(self['/Contents'].get_object(), self.pdf, 'bytes').operations:
        if op == b'TJ':
            s = [x for x in ope[0] if isinstance(x, str)]
        else:
            s = []
        out += op.decode('utf-8') + ' ' + ''.join(s) + ope.__repr__() + '\n'
    out += '\n=============================\n'
    try:
        for fo in self[PG.RESOURCES]['/Font']:
            out += fo + '\n'
            out += self[PG.RESOURCES]['/Font'][fo].__repr__() + '\n'
            try:
                enc_repr = self[PG.RESOURCES]['/Font'][fo]['/Encoding'].__repr__()
                out += enc_repr + '\n'
            except Exception:
                pass
            try:
                out += self[PG.RESOURCES]['/Font'][fo]['/ToUnicode'].get_data().decode() + '\n'
            except Exception:
                pass
    except KeyError:
        out += 'No Font\n'
    return out