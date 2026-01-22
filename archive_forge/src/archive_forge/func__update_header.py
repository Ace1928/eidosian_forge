from __future__ import annotations
from typing import (
import param
from ..models import Card as BkCard
from .base import Column, Row
def _update_header(self, *events):
    from ..pane import HTML, panel
    if self.header is None:
        params = {'object': f'<h3>{self.title}</h3>' if self.title else '&#8203;', 'css_classes': self.title_css_classes, 'margin': (5, 0)}
        if self.header_color:
            params['styles'] = {'color': self.header_color}
        if self._header is not None:
            self._header.param.update(**params)
            return
        else:
            self._header = item = HTML(**params)
    else:
        item = panel(self.header)
        self._header = None
    self._header_layout[:] = [item]