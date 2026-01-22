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
class Chord(Graph):
    """
    Chord is a special type of Graph which computes the locations of
    each node on a circle and the chords connecting them. The amount
    of radial angle devoted to each node and the number of chords are
    scaled by a weight supplied as a value dimension.

    If the values are integers then the number of chords is directly
    scaled by the value, if the values are floats then the number of
    chords are apportioned such that the lowest value edge is given
    one chord and all other nodes are given nodes proportional to
    their weight.
    """
    group = param.String(default='Chord', constant=True)

    def __init__(self, data, kdims=None, vdims=None, compute=True, **params):
        if data is None or (isinstance(data, list) and data == []):
            data = (([], [], []),)
        if isinstance(data, tuple):
            data = data + (None,) * (3 - len(data))
            edges, nodes, edgepaths = data
        else:
            edges, nodes, edgepaths = (data, None, None)
        if nodes is not None:
            if not isinstance(nodes, Dataset):
                if nodes.ndims == 3:
                    nodes = Nodes(nodes)
                else:
                    nodes = Dataset(nodes)
                    nodes = nodes.clone(kdims=nodes.kdims[0], vdims=nodes.kdims[1:])
        super(Graph, self).__init__(edges, kdims=kdims, vdims=vdims, **params)
        if compute:
            self._nodes = nodes
            chord = layout_chords(self)
            self._nodes = chord.nodes
            self._edgepaths = chord.edgepaths
            self._angles = chord._angles
        else:
            if not isinstance(nodes, Nodes):
                raise TypeError(f'Expected Nodes object in data, found {type(nodes)}.')
            self._nodes = nodes
            if not isinstance(edgepaths, EdgePaths):
                raise TypeError('Expected EdgePaths object in data, found %s.' % type(edgepaths))
            self._edgepaths = edgepaths
        self._validate()

    @property
    def edgepaths(self):
        return self._edgepaths

    @property
    def nodes(self):
        return self._nodes