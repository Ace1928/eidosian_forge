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
def transfer_rotation_to_content(self) -> None:
    """
        Apply the rotation of the page to the content and the media/crop/...
        boxes.

        It's recommended to apply this function before page merging.
        """
    r = -self.rotation
    self.rotation = 0
    mb = RectangleObject(self.mediabox)
    trsf = Transformation().translate(-float(mb.left + mb.width / 2), -float(mb.bottom + mb.height / 2)).rotate(r)
    pt1 = trsf.apply_on(mb.lower_left)
    pt2 = trsf.apply_on(mb.upper_right)
    trsf = trsf.translate(-min(pt1[0], pt2[0]), -min(pt1[1], pt2[1]))
    self.add_transformation(trsf, False)
    for b in ['/MediaBox', '/CropBox', '/BleedBox', '/TrimBox', '/ArtBox']:
        if b in self:
            rr = RectangleObject(self[b])
            pt1 = trsf.apply_on(rr.lower_left)
            pt2 = trsf.apply_on(rr.upper_right)
            self[NameObject(b)] = RectangleObject((min(pt1[0], pt2[0]), min(pt1[1], pt2[1]), max(pt1[0], pt2[0]), max(pt1[1], pt2[1])))