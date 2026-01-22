from collections import defaultdict
import numpy as np
import param
from bokeh.models import (
from ...core.data import Dataset
from ...core.options import Cycle, abbreviated_exception
from ...core.util import dimension_sanitizer, unique_array
from ...util.transform import dim
from ..mixins import ChordMixin, GraphMixin
from ..util import get_directed_graph_paths, process_cmap
from .chart import ColorbarPlot, PointPlot
from .element import CompositeElementPlot, LegendPlot
from .styles import (
class ChordPlot(ChordMixin, GraphPlot):
    labels = param.ClassSelector(class_=(str, dim), doc='\n        The dimension or dimension value transform used to draw labels from.')
    show_frame = param.Boolean(default=False, doc='\n        Whether or not to show a complete frame around the plot.')
    label_index = param.ClassSelector(default=None, class_=(str, int), allow_None=True, doc='\n      Index of the dimension from which the node labels will be drawn')
    _style_groups = {'scatter': 'node', 'multi_line': 'edge', 'text': 'label'}
    style_opts = GraphPlot.style_opts + ['label_' + p for p in base_properties + text_properties]
    _draw_order = ['multi_line_2', 'graph', 'text_1']

    def _sync_arcs(self):
        arc_renderer = self.handles['multi_line_2_glyph_renderer']
        scatter_renderer = self.handles['scatter_1_glyph_renderer']
        for gtype in ('selection_', 'nonselection_', 'muted_', 'hover_', ''):
            glyph = getattr(scatter_renderer, gtype + 'glyph')
            arc_glyph = getattr(arc_renderer, gtype + 'glyph')
            if not glyph or not arc_glyph:
                continue
            scatter_props = glyph.properties_with_values(include_defaults=False)
            styles = {k.replace('fill', 'line'): v for k, v in scatter_props.items() if 'fill' in k}
            arc_glyph.update(**styles)

    def _init_glyphs(self, plot, element, ranges, source):
        super()._init_glyphs(plot, element, ranges, source)
        if 'multi_line_2_glyph' in self.handles:
            arc_renderer = self.handles['multi_line_2_glyph_renderer']
            scatter_renderer = self.handles['scatter_1_glyph_renderer']
            arc_renderer.view = scatter_renderer.view
            arc_renderer.data_source = scatter_renderer.data_source
            self.handles['multi_line_2_source'] = scatter_renderer.data_source
            self._sync_arcs()

    def _update_glyphs(self, element, ranges, style):
        if 'multi_line_2_glyph' in self.handles:
            self._sync_arcs()
        super()._update_glyphs(element, ranges, style)

    def get_data(self, element, ranges, style):
        offset = style.pop('label_offset', 1.05)
        data, mapping, style = super().get_data(element, ranges, style)
        angles = element._angles
        arcs = defaultdict(list)
        for i in range(len(element.nodes)):
            start, end = angles[i:i + 2]
            vals = np.linspace(start, end, 20)
            xs, ys = (np.cos(vals), np.sin(vals))
            arcs['arc_xs'].append(xs)
            arcs['arc_ys'].append(ys)
        data['scatter_1'].update(arcs)
        data['multi_line_2'] = data['scatter_1']
        mapping['multi_line_2'] = {'xs': 'arc_xs', 'ys': 'arc_ys', 'line_width': 10}
        label_dim = element.nodes.get_dimension(self.label_index)
        labels = self.labels
        if label_dim and labels:
            self.param.warning("Cannot declare style mapping for 'labels' option and declare a label_index; ignoring the label_index.")
        elif label_dim:
            labels = label_dim
        elif isinstance(labels, str):
            labels = element.nodes.get_dimension(labels)
        if labels is None:
            return (data, mapping, style)
        nodes = element.nodes
        if element.vdims:
            values = element.dimension_values(element.vdims[0])
            if values.dtype.kind in 'uif':
                edges = Dataset(element)[values > 0]
                nodes = list(np.unique([edges.dimension_values(i) for i in range(2)]))
                nodes = element.nodes.select(**{element.nodes.kdims[2].name: nodes})
        xs, ys = (nodes.dimension_values(i) * offset for i in range(2))
        if isinstance(labels, dim):
            text = labels.apply(element, flat=True)
        else:
            text = element.nodes.dimension_values(labels)
            text = [labels.pprint_value(v) for v in text]
        angles = np.arctan2(ys, xs)
        data['text_1'] = dict(x=xs, y=ys, text=[str(l) for l in text], angle=angles)
        mapping['text_1'] = dict(text='text', x='x', y='y', angle='angle', text_baseline='middle')
        return (data, mapping, style)