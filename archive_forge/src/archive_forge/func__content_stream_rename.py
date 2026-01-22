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
def _content_stream_rename(stream: ContentStream, rename: Dict[Any, Any], pdf: Optional[PdfCommonDocProtocol]) -> ContentStream:
    if not rename:
        return stream
    stream = ContentStream(stream, pdf)
    for operands, _operator in stream.operations:
        if isinstance(operands, list):
            for i, op in enumerate(operands):
                if isinstance(op, NameObject):
                    operands[i] = rename.get(op, op)
        elif isinstance(operands, dict):
            for i, op in operands.items():
                if isinstance(op, NameObject):
                    operands[i] = rename.get(op, op)
        else:
            raise KeyError(f'type of operands is {type(operands)}')
    return stream