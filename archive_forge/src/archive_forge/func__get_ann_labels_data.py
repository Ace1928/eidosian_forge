import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
def _get_ann_labels_data(self, order_ann, bins_ann):
    """
        Generate ColumnDataSource dictionary for annular labels.
        """
    if self.yticks is None:
        return dict(x=[], y=[], text=[], angle=[])
    mapping = self._compute_tick_mapping('radius', order_ann, bins_ann)
    values = [(label, radius[0]) for label, radius in mapping.items()]
    labels, radius = zip(*values)
    radius = np.array(radius)
    y_coord = np.sin(np.deg2rad(self.yrotation)) * radius + self.max_radius
    x_coord = np.cos(np.deg2rad(self.yrotation)) * radius + self.max_radius
    return dict(x=x_coord, y=y_coord, text=labels, angle=[0] * len(labels))