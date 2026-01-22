from __future__ import annotations
from collections.abc import Generator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
def _handle_wrapping(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
    """Update figure structure parameters based on facet/pair wrapping."""
    self.wrap = wrap = facet_spec.get('wrap') or pair_spec.get('wrap')
    if not wrap:
        return
    wrap_dim = 'row' if self.subplot_spec['nrows'] > 1 else 'col'
    flow_dim = {'row': 'col', 'col': 'row'}[wrap_dim]
    n_subplots = self.subplot_spec[f'n{wrap_dim}s']
    flow = int(np.ceil(n_subplots / wrap))
    if wrap < self.subplot_spec[f'n{wrap_dim}s']:
        self.subplot_spec[f'n{wrap_dim}s'] = wrap
    self.subplot_spec[f'n{flow_dim}s'] = flow
    self.n_subplots = n_subplots
    self.wrap_dim = wrap_dim