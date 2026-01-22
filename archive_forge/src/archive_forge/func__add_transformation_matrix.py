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
@staticmethod
def _add_transformation_matrix(contents: Any, pdf: Optional[PdfCommonDocProtocol], ctm: CompressedTransformationMatrix) -> ContentStream:
    """Add transformation matrix at the beginning of the given contents stream."""
    a, b, c, d, e, f = ctm
    contents = ContentStream(contents, pdf)
    contents.operations.insert(0, [[FloatObject(a), FloatObject(b), FloatObject(c), FloatObject(d), FloatObject(e), FloatObject(f)], ' cm'])
    return contents