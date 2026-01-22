import sys
from io import BytesIO
from typing import Any, List, Tuple, Union, cast
from ._utils import check_if_whitespace_only, logger_warning
from .constants import ColorSpaces
from .errors import PdfReadError
from .generic import (
def _handle_jpx(size: Tuple[int, int], data: bytes, mode: mode_str_type, color_space: str, colors: int) -> Tuple[Image.Image, str, str, bool]:
    """
    Process image encoded in flateEncode
    Returns img, image_format, extension, inversion
    """
    extension = '.jp2'
    img1 = Image.open(BytesIO(data), formats=('JPEG2000',))
    mode, invert_color = _get_imagemode(color_space, colors, mode)
    if mode == '':
        mode = cast(mode_str_type, img1.mode)
        invert_color = mode in ('CMYK',)
    if img1.mode == 'RGBA' and mode == 'RGB':
        mode = 'RGBA'
    try:
        if img1.mode != mode:
            img = Image.frombytes(mode, img1.size, img1.tobytes())
        else:
            img = img1
    except OSError:
        img = Image.frombytes(mode, img1.size, img1.tobytes())
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    image_format = 'JPEG2000'
    return (img, image_format, extension, invert_color)