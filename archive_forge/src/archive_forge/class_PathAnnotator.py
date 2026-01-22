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
class PathAnnotator(Annotator):
    """
    Annotator which allows drawing and editing Paths and associating
    values with each path and each vertex of a path using a table.
    """
    edit_vertices = param.Boolean(default=True, doc='\n        Whether to add tool to edit vertices.')
    object = param.ClassSelector(class_=Path, doc='\n        Path object to edit and annotate.')
    show_vertices = param.Boolean(default=True, doc='\n        Whether to show vertices when drawing the Path.')
    vertex_annotations = param.ClassSelector(default=[], class_=(dict, list), doc='\n        Columns to annotate the Polygons with.')
    vertex_style = param.Dict(default={'nonselection_alpha': 0.5}, doc='\n        Options to apply to vertices during drawing and editing.')
    _vertex_table_link = VertexTableLink
    _triggers = ['annotations', 'edit_vertices', 'object', 'table_opts', 'vertex_annotations']

    def __init__(self, object=None, **params):
        self._vertex_table_row = Row()
        super().__init__(object, **params)
        self.editor.append((f'{param_name(self.name)} Vertices', self._vertex_table_row))

    def _init_stream(self):
        name = param_name(self.name)
        self._stream = PolyDraw(source=self.plot, data={}, num_objects=self.num_objects, show_vertices=self.show_vertices, tooltip=f'{name} Tool', vertex_style=self.vertex_style, empty_value=self.empty_value)
        if self.edit_vertices:
            self._vertex_stream = PolyEdit(source=self.plot, tooltip=f'{name} Edit Tool', vertex_style=self.vertex_style)

    def _process_element(self, element=None):
        if element is None or not isinstance(element, self._element_type):
            datatype = list(self._element_type.datatype)
            datatype.remove('multitabular')
            datatype.append('multitabular')
            element = self._element_type(element, datatype=datatype)
        validate = []
        for col in self.annotations:
            if col in element:
                validate.append(col)
                continue
            init = self.annotations[col]() if isinstance(self.annotations, dict) else ''
            element = element.add_dimension(col, len(element.vdims), init, True)
        for col in self.vertex_annotations:
            if col in element:
                continue
            elif isinstance(self.vertex_annotations, dict):
                init = self.vertex_annotations[col]()
            else:
                init = ''
            element = element.add_dimension(col, len(element.vdims), init, True)
        poly_data = {c: element.dimension_values(c, expanded=False) for c in validate}
        if validate and len({len(v) for v in poly_data.values()}) != 1:
            raise ValueError('annotations must refer to value dimensions which vary per path while at least one of %s varies by vertex.' % validate)
        tools = [tool() for tool in self._tools]
        opts = dict(tools=tools, color_index=None, **self.default_opts)
        opts.update(self._extra_opts)
        return element.options(**{k: v for k, v in opts.items() if k not in element.opts.get('plot').kwargs})

    def _update_links(self):
        super()._update_links()
        if hasattr(self, '_vertex_link'):
            self._vertex_link.unlink()
        self._vertex_link = self._vertex_table_link(self.plot, self._vertex_table)

    def _update_object(self, data=None):
        if self._stream.element is not None:
            element = self._stream.element
            if element.interface.datatype == 'multitabular' and element.data and isinstance(element.data[0], dict):
                for path in element.data:
                    for col in self.annotations:
                        if not isscalar(path[col]) and len(path[col]):
                            path[col] = path[col][0]
            with param.discard_events(self):
                self.object = element

    def _update_table(self):
        name = param_name(self.name)
        annotations = list(self.annotations)
        table = self.object
        for transform in self.table_transforms:
            table = transform(table)
        table_data = {a: list(table.dimension_values(a, expanded=False)) for a in annotations}
        self._table = Table(table_data, annotations, [], label=name).opts(show_title=False, **self.table_opts)
        self._vertex_table = Table([], table.kdims, list(self.vertex_annotations), label=f'{name} Vertices').opts(show_title=False, **self.table_opts)
        self._update_links()
        self._table_row[:] = [self._table]
        self._vertex_table_row[:] = [self._vertex_table]

    @property
    def selected(self):
        index = self._selection.index
        data = [p for i, p in enumerate(self._stream.element.split()) if i in index]
        return self.object.clone(data)