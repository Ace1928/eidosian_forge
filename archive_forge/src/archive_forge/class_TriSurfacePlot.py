import numpy as np
import param
from plotly import colors
from plotly.figure_factory._trisurf import trisurf as trisurface
from ...core.options import SkipRendering
from .chart import CurvePlot, ScatterPlot
from .element import ColorbarPlot, ElementPlot
from .selection import PlotlyOverlaySelectionDisplay
class TriSurfacePlot(Chart3DPlot, ColorbarPlot):
    style_opts = ['cmap', 'edges_color', 'facecolor']
    selection_display = PlotlyOverlaySelectionDisplay(supports_region=False)

    def get_data(self, element, ranges, style, **kwargs):
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise SkipRendering('SciPy not available, cannot plot TriSurface') from None
        x, y, z = (element.dimension_values(i) for i in range(3))
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)
        simplices = tri.simplices
        return [dict(x=x, y=y, z=z, simplices=simplices)]

    def graph_options(self, element, ranges, style, **kwargs):
        opts = super().graph_options(element, ranges, style, **kwargs)
        copts = self.get_color_opts(element.dimensions()[2], element, ranges, style)
        opts['colormap'] = [tuple((v / 255.0 for v in colors.hex_to_rgb(c))) for _, c in copts['colorscale']]
        opts['scale'] = [l for l, _ in copts['colorscale']]
        opts['show_colorbar'] = self.colorbar
        opts['edges_color'] = style.get('edges_color', 'black')
        opts['plot_edges'] = 'edges_color' in style
        opts['colorbar'] = copts.get('colorbar', None)
        return {k: v for k, v in opts.items() if 'legend' not in k and k != 'name'}

    def init_graph(self, datum, options, index=0, **kwargs):
        colorbar = options.pop('colorbar', None)
        trisurface_traces = trisurface(**dict(datum, **options))
        if colorbar:
            marker_traces = [trace for trace in trisurface_traces if trace.type == 'scatter3d' and trace.mode == 'markers']
            if marker_traces:
                marker_traces[0].marker.colorbar = colorbar
        return {'traces': [t.to_plotly_json() for t in trisurface_traces]}