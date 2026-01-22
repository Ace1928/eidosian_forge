from __future__ import annotations
from collections.abc import Generator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
def _check_dimension_uniqueness(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
    """Reject specs that pair and facet on (or wrap to) same figure dimension."""
    err = None
    facet_vars = facet_spec.get('variables', {})
    if facet_spec.get('wrap') and {'col', 'row'} <= set(facet_vars):
        err = 'Cannot wrap facets when specifying both `col` and `row`.'
    elif pair_spec.get('wrap') and pair_spec.get('cross', True) and (len(pair_spec.get('structure', {}).get('x', [])) > 1) and (len(pair_spec.get('structure', {}).get('y', [])) > 1):
        err = 'Cannot wrap subplots when pairing on both `x` and `y`.'
    collisions = {'x': ['columns', 'rows'], 'y': ['rows', 'columns']}
    for pair_axis, (multi_dim, wrap_dim) in collisions.items():
        if pair_axis not in pair_spec.get('structure', {}):
            continue
        elif multi_dim[:3] in facet_vars:
            err = f'Cannot facet the {multi_dim} while pairing on `{pair_axis}``.'
        elif wrap_dim[:3] in facet_vars and facet_spec.get('wrap'):
            err = f'Cannot wrap the {wrap_dim} while pairing on `{pair_axis}``.'
        elif wrap_dim[:3] in facet_vars and pair_spec.get('wrap'):
            err = f'Cannot wrap the {multi_dim} while faceting the {wrap_dim}.'
    if err is not None:
        raise RuntimeError(err)