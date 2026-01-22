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
def _get_fonts_walk(obj: DictionaryObject, fnt: Set[str], emb: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    Get the set of all fonts and all embedded fonts.

    Args:
        obj: Page resources dictionary
        fnt: font
        emb: embedded fonts

    Returns:
        A tuple (fnt, emb)

    If there is a key called 'BaseFont', that is a font that is used in the document.
    If there is a key called 'FontName' and another key in the same dictionary object
    that is called 'FontFilex' (where x is null, 2, or 3), then that fontname is
    embedded.

    We create and add to two sets, fnt = fonts used and emb = fonts embedded.
    """
    fontkeys = ('/FontFile', '/FontFile2', '/FontFile3')

    def process_font(f: DictionaryObject) -> None:
        nonlocal fnt, emb
        f = cast(DictionaryObject, f.get_object())
        if '/BaseFont' in f:
            fnt.add(cast(str, f['/BaseFont']))
        if '/CharProcs' in f or ('/FontDescriptor' in f and any((x in cast(DictionaryObject, f['/FontDescriptor']) for x in fontkeys))) or ('/DescendantFonts' in f and '/FontDescriptor' in cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object()) and any((x in cast(DictionaryObject, cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object())['/FontDescriptor']) for x in fontkeys))):
            try:
                emb.add(cast(str, f['/BaseFont']))
            except KeyError:
                emb.add('(' + cast(str, f['/Subtype']) + ')')
    if '/DR' in obj and '/Font' in cast(DictionaryObject, obj['/DR']):
        for f in cast(DictionaryObject, cast(DictionaryObject, obj['/DR'])['/Font']):
            process_font(f)
    if '/Resources' in obj:
        if '/Font' in cast(DictionaryObject, obj['/Resources']):
            for f in cast(DictionaryObject, cast(DictionaryObject, obj['/Resources'])['/Font']).values():
                process_font(f)
        if '/XObject' in cast(DictionaryObject, obj['/Resources']):
            for x in cast(DictionaryObject, cast(DictionaryObject, obj['/Resources'])['/XObject']).values():
                _get_fonts_walk(cast(DictionaryObject, x.get_object()), fnt, emb)
    if '/Annots' in obj:
        for a in cast(ArrayObject, obj['/Annots']):
            _get_fonts_walk(cast(DictionaryObject, a.get_object()), fnt, emb)
    if '/AP' in obj:
        if cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']).get('/Type') == '/XObject':
            _get_fonts_walk(cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']), fnt, emb)
        else:
            for a in cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']):
                _get_fonts_walk(cast(DictionaryObject, a), fnt, emb)
    return (fnt, emb)