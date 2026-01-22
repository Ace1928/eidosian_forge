from typing import Literal, Optional, Union, cast, Tuple
from .deprecation import AltairDeprecationWarning
from .html import spec_to_html
from ._importers import import_vl_convert, vl_version_for_vl_convert
import struct
import warnings
def _pngxy(data):
    """read the (width, height) from a PNG header

    Taken from IPython.display
    """
    ihdr = data.index(b'IHDR')
    return struct.unpack('>ii', data[ihdr + 4:ihdr + 12])