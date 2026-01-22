from collections import defaultdict
from types import FunctionType
import numpy as np
import pandas as pd
import param
from ..core import Dataset, Dimension, Element2D
from ..core.accessors import Redim
from ..core.operation import Operation
from ..core.util import is_dataframe, max_range, search_indices
from .chart import Points
from .path import Path
from .util import (
def _initialize_edgepaths(self):
    """
        Returns the EdgePaths by generating a triangle for each simplex.
        """
    if self._edgepaths:
        return self._edgepaths
    elif not len(self):
        edgepaths = self.edge_type([], kdims=self.nodes.kdims[:2])
        self._edgepaths = edgepaths
        return edgepaths
    df = connect_tri_edges_pd(self)
    pts = df.values.reshape((len(df), 3, 2))
    paths = np.pad(pts[:, [0, 1, 2, 0], :].astype(float), pad_width=((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=np.nan).reshape(-1, 2)[:-1]
    edgepaths = self.edge_type([paths], kdims=self.nodes.kdims[:2], datatype=['multitabular'])
    self._edgepaths = edgepaths
    return edgepaths