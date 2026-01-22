from __future__ import annotations
import sys
from io import BytesIO
from . import Image
from ._util import is_path
def _toqclass_helper(im):
    data = None
    colortable = None
    exclusive_fp = False
    if hasattr(im, 'toUtf8'):
        im = str(im.toUtf8(), 'utf-8')
    if is_path(im):
        im = Image.open(im)
        exclusive_fp = True
    qt_format = QImage.Format if qt_version == '6' else QImage
    if im.mode == '1':
        format = qt_format.Format_Mono
    elif im.mode == 'L':
        format = qt_format.Format_Indexed8
        colortable = [rgb(i, i, i) for i in range(256)]
    elif im.mode == 'P':
        format = qt_format.Format_Indexed8
        palette = im.getpalette()
        colortable = [rgb(*palette[i:i + 3]) for i in range(0, len(palette), 3)]
    elif im.mode == 'RGB':
        im = im.convert('RGBA')
        data = im.tobytes('raw', 'BGRA')
        format = qt_format.Format_RGB32
    elif im.mode == 'RGBA':
        data = im.tobytes('raw', 'BGRA')
        format = qt_format.Format_ARGB32
    elif im.mode == 'I;16' and hasattr(qt_format, 'Format_Grayscale16'):
        im = im.point(lambda i: i * 256)
        format = qt_format.Format_Grayscale16
    else:
        if exclusive_fp:
            im.close()
        msg = f'unsupported image mode {repr(im.mode)}'
        raise ValueError(msg)
    size = im.size
    __data = data or align8to32(im.tobytes(), size[0], im.mode)
    if exclusive_fp:
        im.close()
    return {'data': __data, 'size': size, 'format': format, 'colortable': colortable}