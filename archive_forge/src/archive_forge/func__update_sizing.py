from __future__ import annotations
from typing import ClassVar, List, Mapping
import param
from ..config import config
from ..io.resources import CDN_DIST, bundled_files
from ..reactive import ReactiveHTML
from ..util import classproperty
from .grid import GridSpec
@param.depends('objects', watch=True)
def _update_sizing(self):
    if self.ncols and self.width:
        width = self.width / self.ncols
    else:
        width = 0
    if self.nrows and self.height:
        height = self.height / self.nrows
    else:
        height = 0
    for (y0, x0, y1, x1), obj in self.objects.items():
        x0 = 0 if x0 is None else x0
        x1 = self.ncols if x1 is None else x1
        y0 = 0 if y0 is None else y0
        y1 = self.nrows if y1 is None else y1
        h, w = (y1 - y0, x1 - x0)
        properties = {}
        if self.sizing_mode in ['fixed', None]:
            if width:
                properties['width'] = int(w * width)
            if height:
                properties['height'] = int(h * height)
        else:
            properties['sizing_mode'] = self.sizing_mode
            if 'width' in self.sizing_mode and height:
                properties['height'] = int(h * height)
            elif 'height' in self.sizing_mode and width:
                properties['width'] = int(w * width)
        obj.param.update(**{k: v for k, v in properties.items() if not obj.param[k].readonly})