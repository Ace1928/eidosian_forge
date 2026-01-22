import sys
from inspect import getmro
import param
from panel.layout import Row, Tabs
from panel.pane import PaneBase
from panel.util import param_name
from .core import DynamicMap, Element, HoloMap, Layout, Overlay, Store, ViewableElement
from .core.util import isscalar
from .element import Curve, Path, Points, Polygons, Rectangles, Table
from .plotting.links import (
from .streams import BoxEdit, CurveEdit, PointDraw, PolyDraw, PolyEdit, Selection1D
class _GeomAnnotator(Annotator):
    default_opts = param.Dict(default={'responsive': True, 'min_height': 400, 'padding': 0.1, 'framewise': True}, doc='\n        Opts to apply to the element.')
    _stream_type = None
    __abstract = True

    def _init_stream(self):
        name = param_name(self.name)
        self._stream = self._stream_type(source=self.plot, data={}, num_objects=self.num_objects, tooltip=f'{name} Tool', empty_value=self.empty_value)

    def _process_element(self, object):
        if object is None or not isinstance(object, self._element_type):
            object = self._element_type(object)
        for col in self.annotations:
            if col in object:
                continue
            init = self.annotations[col]() if isinstance(self.annotations, dict) else ''
            object = object.add_dimension(col, len(object.vdims), init, True)
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, **self.default_opts)
        opts.update(self._extra_opts)
        return object.options(**{k: v for k, v in opts.items() if k not in object.opts.get('plot').kwargs})