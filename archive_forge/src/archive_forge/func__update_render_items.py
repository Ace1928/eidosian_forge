import json
import math
import pathlib
import param
from ...config import config
from ...depends import depends
from ...io.resources import CSS_URLS
from ...layout import GridSpec
from ..base import BasicTemplate
def _update_render_items(self, event):
    super()._update_render_items(event)
    if event.obj is not self.main:
        return
    layouts = []
    for i, ((y0, x0, y1, x1), v) in enumerate(self.main.objects.items()):
        if x0 is None:
            x0 = 0
        if x1 is None:
            x1 = 12
        if y0 is None:
            y0 = 0
        if y1 is None:
            y1 = self.main.nrows
        elem = {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0, 'i': str(i + 1)}
        elem.update({d: v for d, v in self.dimensions.items() if not math.isinf(v)})
        layouts.append(elem)
    self._render_variables['layouts'] = json.dumps({'lg': layouts, 'md': layouts})