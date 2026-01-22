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
def _merge_page_writer(self, page2: 'PageObject', page2transformation: Optional[Callable[[Any], ContentStream]]=None, ctm: Optional[CompressedTransformationMatrix]=None, over: bool=True, expand: bool=False) -> None:
    assert isinstance(self.indirect_reference, IndirectObject)
    pdf = self.indirect_reference.pdf
    rename = {}
    if PG.RESOURCES not in self:
        self[NameObject(PG.RESOURCES)] = DictionaryObject()
    original_resources = cast(DictionaryObject, self[PG.RESOURCES].get_object())
    if PG.RESOURCES not in page2:
        page2resources = DictionaryObject()
    else:
        page2resources = cast(DictionaryObject, page2[PG.RESOURCES].get_object())
    for res in (RES.EXT_G_STATE, RES.FONT, RES.XOBJECT, RES.COLOR_SPACE, RES.PATTERN, RES.SHADING, RES.PROPERTIES):
        if res in page2resources:
            if res not in original_resources:
                original_resources[NameObject(res)] = DictionaryObject()
            _, newrename = self._merge_resources(original_resources, page2resources, res, False)
            rename.update(newrename)
    if RES.PROC_SET in page2resources:
        if RES.PROC_SET not in original_resources:
            original_resources[NameObject(RES.PROC_SET)] = ArrayObject()
        arr = cast(ArrayObject, original_resources[RES.PROC_SET])
        for x in cast(ArrayObject, page2resources[RES.PROC_SET]):
            if x not in arr:
                arr.append(x)
        arr.sort()
    if PG.ANNOTS in page2:
        if PG.ANNOTS not in self:
            self[NameObject(PG.ANNOTS)] = ArrayObject()
        annots = cast(ArrayObject, self[PG.ANNOTS].get_object())
        if ctm is None:
            trsf = Transformation()
        else:
            trsf = Transformation(ctm)
        for a in cast(ArrayObject, page2[PG.ANNOTS]):
            a = a.get_object()
            aa = a.clone(pdf, ignore_fields=('/P', '/StructParent', '/Parent'), force_duplicate=True)
            r = cast(ArrayObject, a['/Rect'])
            pt1 = trsf.apply_on((r[0], r[1]), True)
            pt2 = trsf.apply_on((r[2], r[3]), True)
            aa[NameObject('/Rect')] = ArrayObject((min(pt1[0], pt2[0]), min(pt1[1], pt2[1]), max(pt1[0], pt2[0]), max(pt1[1], pt2[1])))
            if '/QuadPoints' in a:
                q = cast(ArrayObject, a['/QuadPoints'])
                aa[NameObject('/QuadPoints')] = ArrayObject(trsf.apply_on((q[0], q[1]), True) + trsf.apply_on((q[2], q[3]), True) + trsf.apply_on((q[4], q[5]), True) + trsf.apply_on((q[6], q[7]), True))
            try:
                aa['/Popup'][NameObject('/Parent')] = aa.indirect_reference
            except KeyError:
                pass
            try:
                aa[NameObject('/P')] = self.indirect_reference
                annots.append(aa.indirect_reference)
            except AttributeError:
                pass
    new_content_array = ArrayObject()
    original_content = self.get_contents()
    if original_content is not None:
        original_content.isolate_graphics_state()
        new_content_array.append(original_content)
    page2content = page2.get_contents()
    if page2content is not None:
        rect = getattr(page2, MERGE_CROP_BOX)
        page2content.operations.insert(0, (map(FloatObject, [rect.left, rect.bottom, rect.width, rect.height]), 're'))
        page2content.operations.insert(1, ([], 'W'))
        page2content.operations.insert(2, ([], 'n'))
        if page2transformation is not None:
            page2content = page2transformation(page2content)
        page2content = PageObject._content_stream_rename(page2content, rename, self.pdf)
        page2content.isolate_graphics_state()
        if over:
            new_content_array.append(page2content)
        else:
            new_content_array.insert(0, page2content)
    if expand:
        self._expand_mediabox(page2, ctm)
    self.replace_contents(new_content_array)