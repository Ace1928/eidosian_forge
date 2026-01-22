import sys
from io import BytesIO
from typing import Any, List, Tuple, Union, cast
from ._utils import check_if_whitespace_only, logger_warning
from .constants import ColorSpaces
from .errors import PdfReadError
from .generic import (
def _handle_flate(size: Tuple[int, int], data: bytes, mode: mode_str_type, color_space: str, colors: int, obj_as_text: str) -> Tuple[Image.Image, str, str, bool]:
    """
    Process image encoded in flateEncode
    Returns img, image_format, extension, color inversion
    """

    def bits2byte(data: bytes, size: Tuple[int, int], bits: int) -> bytes:
        mask = (2 << bits) - 1
        nbuff = bytearray(size[0] * size[1])
        by = 0
        bit = 8 - bits
        for y in range(size[1]):
            if bit != 0 and bit != 8 - bits:
                by += 1
                bit = 8 - bits
            for x in range(size[0]):
                nbuff[y * size[0] + x] = data[by] >> bit & mask
                bit -= bits
                if bit < 0:
                    by += 1
                    bit = 8 - bits
        return bytes(nbuff)
    extension = '.png'
    image_format = 'PNG'
    lookup: Any
    base: Any
    hival: Any
    if isinstance(color_space, ArrayObject) and color_space[0] == '/Indexed':
        color_space, base, hival, lookup = (value.get_object() for value in color_space)
    if mode == '2bits':
        mode = 'P'
        data = bits2byte(data, size, 2)
    elif mode == '4bits':
        mode = 'P'
        data = bits2byte(data, size, 4)
    img = Image.frombytes(mode, size, data)
    if color_space == '/Indexed':
        from .generic import TextStringObject
        if isinstance(lookup, (EncodedStreamObject, DecodedStreamObject)):
            lookup = lookup.get_data()
        if isinstance(lookup, TextStringObject):
            lookup = lookup.original_bytes
        if isinstance(lookup, str):
            lookup = lookup.encode()
        try:
            nb, conv, mode = {'1': (0, '', ''), 'L': (1, 'P', 'L'), 'P': (0, '', ''), 'RGB': (3, 'P', 'RGB'), 'CMYK': (4, 'P', 'CMYK')}[_get_imagemode(base, 0, '')[0]]
        except KeyError:
            logger_warning(f'Base {base} not coded please share the pdf file with pypdf dev team', __name__)
            lookup = None
        else:
            if img.mode == '1':
                expected_count = 2 * nb
                if len(lookup) != expected_count:
                    if len(lookup) < expected_count:
                        raise PdfReadError(f'Not enough lookup values: Expected {expected_count}, got {len(lookup)}.')
                    if not check_if_whitespace_only(lookup[expected_count:]):
                        raise PdfReadError(f'Too many lookup values: Expected {expected_count}, got {len(lookup)}.')
                    lookup = lookup[:expected_count]
                colors_arr = [lookup[:nb], lookup[nb:]]
                arr = b''.join([b''.join([colors_arr[1 if img.getpixel((x, y)) > 127 else 0] for x in range(img.size[0])]) for y in range(img.size[1])])
                img = Image.frombytes(mode, img.size, arr)
            else:
                img = img.convert(conv)
                if len(lookup) != (hival + 1) * nb:
                    logger_warning(f'Invalid Lookup Table in {obj_as_text}', __name__)
                    lookup = None
                elif mode == 'L':
                    lookup = b''.join([bytes([b, b, b]) for b in lookup])
                    mode = 'RGB'
                elif mode == 'CMYK':
                    _rgb = []
                    for _c, _m, _y, _k in (lookup[n:n + 4] for n in range(0, 4 * (len(lookup) // 4), 4)):
                        _r = int(255 * (1 - _c / 255) * (1 - _k / 255))
                        _g = int(255 * (1 - _m / 255) * (1 - _k / 255))
                        _b = int(255 * (1 - _y / 255) * (1 - _k / 255))
                        _rgb.append(bytes((_r, _g, _b)))
                    lookup = b''.join(_rgb)
                    mode = 'RGB'
                if lookup is not None:
                    img.putpalette(lookup, rawmode=mode)
            img = img.convert('L' if base == ColorSpaces.DEVICE_GRAY else 'RGB')
    elif not isinstance(color_space, NullObject) and color_space[0] == '/ICCBased':
        mode2 = _get_imagemode(color_space, colors, mode)[0]
        if mode != mode2:
            img = Image.frombytes(mode2, size, data)
    if mode == 'CMYK':
        extension = '.tif'
        image_format = 'TIFF'
    return (img, image_format, extension, False)