from collections import defaultdict
from functools import partial
import numpy as np
import param
from bokeh.models import Circle, FactorRange, HBar, VBar
from ...core import NdOverlay
from ...core.dimension import Dimension, Dimensioned
from ...core.ndmapping import sorted_context
from ...core.util import (
from ...operation.stats import univariate_kde
from ...util.transform import dim
from ..mixins import MultiDistributionMixin
from .chart import AreaPlot
from .element import ColorbarPlot, CompositeElementPlot, LegendPlot
from .path import PolygonPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties
from .util import decode_bytes
def _kde_data(self, element, el, key, split_dim, split_cats, **kwargs):
    vdims = el.vdims
    vdim = vdims[0]
    if self.clip:
        vdim = vdim(range=self.clip)
        el = el.clone(vdims=[vdim])
    if split_dim is not None:
        el = el.clone(kdims=element.kdims)
        all_cats = split_dim.apply(el)
        if len(split_cats) > 2:
            raise ValueError('The number of categories for split violin plots cannot be greater than 2. Found {} categories: {}'.format(len(split_cats), ', '.join(split_cats)))
        el = el.add_dimension(repr(split_dim), len(el.kdims), all_cats)
        kdes = univariate_kde(el, dimension=vdim.name, groupby=repr(split_dim), **kwargs)
        scale = 4
    else:
        split_cats = [None, None]
        kdes = {None: univariate_kde(el, dimension=vdim.name, **kwargs)}
        scale = 2
    x_range = el.range(vdim)
    xs, fill_xs, ys, fill_ys = ([], [], [], [])
    for i, cat in enumerate(split_cats):
        kde = kdes.get(cat)
        if kde is None:
            _xs, _ys = (np.array([]), np.array([]))
        else:
            _xs, _ys = (kde.dimension_values(idim) for idim in range(2))
        mask = isfinite(_ys) & (_ys > 0)
        _xs, _ys = (_xs[mask], _ys[mask])
        if i == 0:
            _ys *= -1
        else:
            _ys = _ys[::-1]
            _xs = _xs[::-1]
        if split_dim:
            if len(_xs):
                fill_xs.append([x_range[0]] + list(_xs) + [x_range[-1]])
                fill_ys.append([0] + list(_ys) + [0])
            else:
                fill_xs.append([])
                fill_ys.append([])
        x_range = x_range[::-1]
        xs += list(_xs)
        ys += list(_ys)
    xs = np.array(xs)
    ys = np.array(ys)
    if split_dim:
        fill_xs = [np.asarray(x) for x in fill_xs]
        fill_ys = [[key + (y,) for y in fy / np.abs(ys).max() * (self.violin_width / scale)] if len(fy) else [] for fy in fill_ys]
    ys = ys / np.nanmax(np.abs(ys)) * (self.violin_width / scale) if len(ys) else []
    ys = [key + (y,) for y in ys]
    line = {'ys': xs, 'xs': ys}
    if split_dim:
        kde = {'ys': fill_xs, 'xs': fill_ys}
    else:
        kde = line
    if isinstance(kdes, NdOverlay):
        kde[repr(split_dim)] = [str(k) for k in split_cats]
    bars, segments, scatter = (defaultdict(list), defaultdict(list), {})
    values = el.dimension_values(vdim)
    values = values[isfinite(values)]
    if not len(values):
        pass
    elif self.inner == 'quartiles':
        if len(xs):
            for stat_fn in self._stat_fns:
                stat = stat_fn(values)
                sidx = np.argmin(np.abs(xs - stat))
                sx, sy = (xs[sidx], ys[sidx])
                segments['x'].append(sx)
                segments['y0'].append(key + (-sy[-1],))
                segments['y1'].append(sy)
    elif self.inner == 'stick':
        if len(xs):
            for value in values:
                sidx = np.argmin(np.abs(xs - value))
                sx, sy = (xs[sidx], ys[sidx])
                segments['x'].append(sx)
                segments['y0'].append(key + (-sy[-1],))
                segments['y1'].append(sy)
    elif self.inner == 'box':
        xpos = key + (0,)
        q1, q2, q3, upper, lower, _ = self._box_stats(values)
        segments['x'].append(xpos)
        segments['y0'].append(lower)
        segments['y1'].append(upper)
        bars['x'].append(xpos)
        bars['bottom'].append(q1)
        bars['top'].append(q3)
        scatter['x'] = xpos
        scatter['y'] = q2
    return (kde, line, segments, bars, scatter)