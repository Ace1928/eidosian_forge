import os
import os.path
import struct
from io import BytesIO
from typing import BinaryIO, Tuple
from .jbig2 import JBIG2StreamReader, JBIG2StreamWriter
from .layout import LTImage
from .pdfcolor import LITERAL_DEVICE_CMYK
from .pdfcolor import LITERAL_DEVICE_GRAY
from .pdfcolor import LITERAL_DEVICE_RGB
from .pdftypes import (
@staticmethod
def _is_jbig2_iamge(image: LTImage) -> bool:
    filters = image.stream.get_filters()
    for filter_name, params in filters:
        if filter_name in LITERALS_JBIG2_DECODE:
            return True
    return False