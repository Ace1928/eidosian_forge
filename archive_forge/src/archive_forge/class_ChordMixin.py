import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class ChordMixin:

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        """
        A Chord plot is always drawn on a unit circle.
        """
        xdim, ydim = element.nodes.kdims[:2]
        if range_type not in ('combined', 'data', 'extents'):
            return (xdim.range[0], ydim.range[0], xdim.range[1], ydim.range[1])
        no_labels = element.nodes.get_dimension(self.label_index) is None and self.labels is None
        rng = 1.1 if no_labels else 1.4
        x0, x1 = util.max_range([xdim.range, (-rng, rng)])
        y0, y1 = util.max_range([ydim.range, (-rng, rng)])
        return (x0, y0, x1, y1)