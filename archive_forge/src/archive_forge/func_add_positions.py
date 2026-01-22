import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def add_positions(self, position):
    """Add one or more events at the specified positions."""
    if position is None or (hasattr(position, 'len') and len(position) == 0):
        return
    positions = self.get_positions()
    positions = np.hstack([positions, np.asanyarray(position)])
    self.set_positions(positions)