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
def _set_interaction_policies(self, renderer):
    if self.selection_policy == 'nodes':
        renderer.selection_policy = NodesAndLinkedEdges()
    elif self.selection_policy == 'edges':
        renderer.selection_policy = EdgesAndLinkedNodes()
    else:
        renderer.selection_policy = NodesOnly()
    if self.inspection_policy == 'nodes':
        renderer.inspection_policy = NodesAndLinkedEdges()
    elif self.inspection_policy == 'edges':
        renderer.inspection_policy = EdgesAndLinkedNodes()
    else:
        renderer.inspection_policy = NodesOnly()