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

            Find a key that either doesn't already exist or has the same value
            (indicated by the bool)

            Args:
                base_key: An index is added to this to get the computed key

            Returns:
                A tuple (computed key, bool) where the boolean indicates
                if there is a resource of the given computed_key with the same
                value.
            