from __future__ import annotations
from collections.abc import Generator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
    """Update subplot spec with default or specified axis sharing parameters."""
    axis_to_dim = {'x': 'col', 'y': 'row'}
    key: str
    val: str | bool
    for axis in 'xy':
        key = f'share{axis}'
        if key not in self.subplot_spec:
            if axis in pair_spec.get('structure', {}):
                if self.wrap is None and pair_spec.get('cross', True):
                    val = axis_to_dim[axis]
                else:
                    val = False
            else:
                val = True
            self.subplot_spec[key] = val