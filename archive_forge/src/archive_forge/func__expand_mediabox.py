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
def _expand_mediabox(self, page2: 'PageObject', ctm: Optional[CompressedTransformationMatrix]) -> None:
    corners1 = (self.mediabox.left.as_numeric(), self.mediabox.bottom.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.top.as_numeric())
    corners2 = (page2.mediabox.left.as_numeric(), page2.mediabox.bottom.as_numeric(), page2.mediabox.left.as_numeric(), page2.mediabox.top.as_numeric(), page2.mediabox.right.as_numeric(), page2.mediabox.top.as_numeric(), page2.mediabox.right.as_numeric(), page2.mediabox.bottom.as_numeric())
    if ctm is not None:
        ctm = tuple((float(x) for x in ctm))
        new_x = tuple((ctm[0] * corners2[i] + ctm[2] * corners2[i + 1] + ctm[4] for i in range(0, 8, 2)))
        new_y = tuple((ctm[1] * corners2[i] + ctm[3] * corners2[i + 1] + ctm[5] for i in range(0, 8, 2)))
    else:
        new_x = corners2[0:8:2]
        new_y = corners2[1:8:2]
    lowerleft = (min(new_x), min(new_y))
    upperright = (max(new_x), max(new_y))
    lowerleft = (min(corners1[0], lowerleft[0]), min(corners1[1], lowerleft[1]))
    upperright = (max(corners1[2], upperright[0]), max(corners1[3], upperright[1]))
    self.mediabox.lower_left = lowerleft
    self.mediabox.upper_right = upperright