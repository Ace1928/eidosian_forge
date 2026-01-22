import param
from holoviews.plotting.util import attach_streams
from ...core import AdjointLayout, Empty, GridMatrix, GridSpace, HoloMap, NdLayout
from ...core.options import Store
from ...core.util import wrap_tuple
from ...element import Histogram
from ..plot import (
from .util import configure_matching_axes_from_dims, figure_grid
def _create_subplots(self, layout, ranges):
    subplots = {}
    frame_ranges = self.compute_ranges(layout, None, ranges)
    frame_ranges = dict([(key, self.compute_ranges(layout, key, frame_ranges)) for key in self.keys])
    collapsed_layout = layout.clone(shared_data=False, id=layout.id)
    for coord in layout.keys(full_grid=True):
        if not isinstance(coord, tuple):
            coord = (coord,)
        view = layout.data.get(coord, None)
        if view is not None:
            vtype = view.type if isinstance(view, HoloMap) else view.__class__
            opts = self.lookup_options(view, 'plot').options
        else:
            vtype = None
        kwargs = {}
        if isinstance(layout, GridMatrix):
            if view.traverse(lambda x: x, [Histogram]):
                kwargs['shared_axes'] = False
        plotting_class = Store.registry[self.renderer.backend].get(vtype, None)
        if plotting_class is None:
            if view is not None:
                self.param.warning('Plotly plotting class for %s type not found, object will not be rendered.' % vtype.__name__)
        else:
            subplot = plotting_class(view, dimensions=self.dimensions, show_title=False, subplot=True, ranges=frame_ranges, uniform=self.uniform, keys=self.keys, **dict(opts, **kwargs))
            collapsed_layout[coord] = subplot.layout if isinstance(subplot, GenericCompositePlot) else subplot.hmap
            subplots[coord] = subplot
    return (subplots, collapsed_layout)