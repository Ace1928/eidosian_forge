from __future__ import annotations
import sys
from io import BytesIO
from . import Image
from ._util import is_path

            An PIL image wrapper for Qt.  This is a subclass of PyQt's QImage
            class.

            :param im: A PIL Image object, or a file name (given either as
                Python string or a PyQt string object).
            