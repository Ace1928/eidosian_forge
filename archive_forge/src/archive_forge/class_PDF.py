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
class PDF(FileBase):
    """
    The `PDF` pane embeds a .pdf image file in a panel if provided a
    local path, or will link to a remote image if provided a URL.

    Reference: https://panel.holoviz.org/reference/panes/PDF.html

    :Example:

    >>> PDF(
    ...     'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
    ...     width=300, height=410
    ... )
    """
    start_page = param.Integer(default=1, doc='\n        Start page of the pdf, by default the first page.')
    filetype: ClassVar[str] = 'pdf'
    _bokeh_model: ClassVar[Model] = _BkPDF
    _rename: ClassVar[Mapping[str, str | None]] = {'embed': 'embed'}
    _rerender_params: ClassVar[List[str]] = FileBase._rerender_params + ['start_page']

    def _transform_object(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return dict(object='<embed></embed>')
        elif self.embed or not isurl(obj):
            data = self._data(obj)
            if not isinstance(data, bytes):
                data = data.encode('utf-8')
            b64 = base64.b64encode(data).decode('utf-8')
            if self.embed:
                return dict(text=b64)
            obj = f'data:application/pdf;base64,{b64}'
        w, h = (self.width or '100%', self.height or '100%')
        page = f'#page={self.start_page}' if getattr(self, 'start_page', None) else ''
        html = f'<embed src="{obj}{page}" width={w!r} height={h!r} type="application/pdf">'
        return dict(text=escape(html))