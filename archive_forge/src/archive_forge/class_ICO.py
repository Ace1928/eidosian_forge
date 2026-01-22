from __future__ import annotations
import asyncio
import base64
import struct
from io import BytesIO
from pathlib import PurePath
from typing import (
import param
from ..models import PDF as _BkPDF
from ..util import isfile, isurl
from .markup import HTMLBasePane, escape
class ICO(ImageBase):
    """
    The `ICO` pane embeds an .ico image file in a panel if provided a local
    path, or will link to a remote image if provided a URL.

    Reference: https://panel.holoviz.org/reference/panes/ICO.html

    :Example:

    >>> ICO(
    ...     some_url,
    ...     alt_text='An .ico file',
    ...     link_url='https://en.wikipedia.org/wiki/ICO_(file_format)',
    ...     width=50
    ...
    """
    filetype: ClassVar[str] = 'ico'

    @classmethod
    def _imgshape(cls, data):
        import struct
        w, h = struct.unpack('<BB', data[6:8])
        return (int(w or 256), int(h or 256))