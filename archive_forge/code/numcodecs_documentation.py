from __future__ import annotations
from io import BytesIO
from typing import TYPE_CHECKING, Any
from numcodecs import registry
from numcodecs.abc import Codec
from .tifffile import TiffFile, TiffWriter
Return decoded image as NumPy array.