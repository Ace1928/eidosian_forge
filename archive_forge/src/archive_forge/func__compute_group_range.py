import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
@classmethod
def _compute_group_range(cls, group, elements, ranges, framewise, axiswise, robust, top_level, prev_frame):
    elements = [el for el in elements if el is not None]
    data_ranges = {}
    robust_ranges = {}
    categorical_dims = []
    for el in elements:
        for el_dim in el.dimensions('ranges'):
            if hasattr(el, 'interface'):
                if isinstance(el, Graph) and el_dim in el.nodes.dimensions():
                    dtype = el.nodes.interface.dtype(el.nodes, el_dim)
                else:
                    dtype = el.interface.dtype(el, el_dim)
            elif hasattr(el, '__len__') and len(el):
                dtype = el.dimension_values(el_dim).dtype
            else:
                dtype = None
            if all((util.isfinite(r) for r in el_dim.range)):
                data_range = (None, None)
            elif dtype is not None and dtype.kind in 'SU':
                data_range = ('', '')
            elif isinstance(el, Graph) and el_dim in el.kdims[:2]:
                data_range = el.nodes.range(2, dimension_range=False)
            elif el_dim.values:
                ds = Dataset(el_dim.values, el_dim)
                data_range = ds.range(el_dim, dimension_range=False)
            else:
                data_range = el.range(el_dim, dimension_range=False)
            data_ranges[el, el_dim] = data_range
            if dtype is not None and dtype.kind in 'uif' and robust:
                percentile = 2 if isinstance(robust, bool) else robust
                robust_ranges[el, el_dim] = (dim(el_dim, np.nanpercentile, percentile).apply(el), dim(el_dim, np.nanpercentile, 100 - percentile).apply(el))
            if any((isinstance(r, str) for r in data_range)) or (el_dim.type is not None and issubclass(el_dim.type, str)) or (dtype is not None and dtype.kind in 'SU'):
                categorical_dims.append(el_dim)
    prev_ranges = ranges.get(group, {})
    group_ranges = {}
    for el in elements:
        if isinstance(el, (Empty, Table)):
            continue
        opts = cls.lookup_options(el, 'style')
        plot_opts = cls.lookup_options(el, 'plot')
        opt_kwargs = dict(opts.kwargs, **plot_opts.kwargs)
        if not opt_kwargs.get('apply_ranges', True):
            continue
        for k, v in opt_kwargs.items():
            if not isinstance(v, dim) or ('color' not in k and k != 'magnitude'):
                continue
            if isinstance(v, dim) and v.applies(el):
                dim_name = repr(v)
                if dim_name in prev_ranges and (not framewise):
                    continue
                values = v.apply(el, all_values=True)
                factors = None
                if values.dtype.kind == 'M':
                    drange = (values.min(), values.max())
                elif util.isscalar(values):
                    drange = (values, values)
                elif values.dtype.kind in 'US':
                    factors = util.unique_array(values)
                elif len(values) == 0:
                    drange = (np.nan, np.nan)
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
                            drange = (np.nanmin(values), np.nanmax(values))
                    except Exception:
                        factors = util.unique_array(values)
                if dim_name not in group_ranges:
                    group_ranges[dim_name] = {'id': [], 'data': [], 'hard': [], 'soft': []}
                if factors is not None:
                    if 'factors' not in group_ranges[dim_name]:
                        group_ranges[dim_name]['factors'] = []
                    group_ranges[dim_name]['factors'].append(factors)
                else:
                    group_ranges[dim_name]['data'].append(drange)
                group_ranges[dim_name]['id'].append(id(el))
        for el_dim in el.dimensions('ranges'):
            dim_name = el_dim.name
            if dim_name in prev_ranges and (not framewise):
                continue
            data_range = data_ranges[el, el_dim]
            if dim_name not in group_ranges:
                group_ranges[dim_name] = {'id': [], 'data': [], 'hard': [], 'soft': [], 'robust': []}
            group_ranges[dim_name]['data'].append(data_range)
            group_ranges[dim_name]['hard'].append(el_dim.range)
            group_ranges[dim_name]['soft'].append(el_dim.soft_range)
            if (el, el_dim) in robust_ranges:
                group_ranges[dim_name]['robust'].append(robust_ranges[el, el_dim])
            if el_dim in categorical_dims:
                if 'factors' not in group_ranges[dim_name]:
                    group_ranges[dim_name]['factors'] = []
                if el_dim.values not in ([], None):
                    values = el_dim.values
                elif el_dim in el:
                    if isinstance(el, Graph) and el_dim in el.kdims[:2]:
                        values = el.nodes.dimension_values(2, expanded=False)
                    else:
                        values = el.dimension_values(el_dim, expanded=False)
                elif isinstance(el, Graph) and el_dim in el.nodes:
                    values = el.nodes.dimension_values(el_dim, expanded=False)
                if isinstance(values, np.ndarray) and values.dtype.kind == 'O' and all((isinstance(v, np.ndarray) for v in values)):
                    values = np.concatenate(values) if len(values) else []
                factors = util.unique_array(values)
                group_ranges[dim_name]['factors'].append(factors)
            group_ranges[dim_name]['id'].append(id(el))
    group_dim_ranges = defaultdict(dict)
    for gdim, values in group_ranges.items():
        matching = True
        for t, rs in values.items():
            if t in ('factors', 'id'):
                continue
            matching &= len({'date' if isinstance(v, util.datetime_types) else 'number' for rng in rs for v in rng if util.isfinite(v)}) < 2
        if matching:
            group_dim_ranges[gdim] = values
    dim_ranges = []
    for gdim, values in group_dim_ranges.items():
        dranges = cls._merge_group_ranges(values)
        dim_ranges.append((gdim, dranges))
    if prev_ranges and (not (top_level or axiswise)) and framewise and (prev_frame is not None):
        prev_ids = prev_frame.traverse(lambda o: id(o))
        for d, dranges in dim_ranges:
            values = prev_ranges.get(d, {}).get('values', None)
            if values is None or 'id' not in values:
                for g, drange in dranges.items():
                    if d not in prev_ranges:
                        prev_ranges[d] = {}
                    prev_ranges[d][g] = drange
                continue
            ids = values.get('id')
            merged = {}
            for g, drange in dranges['values'].items():
                filtered = [r for i, r in zip(ids, values.get(g, [])) if i not in prev_ids]
                filtered += drange
                merged[g] = filtered
            prev_ranges[d] = cls._merge_group_ranges(merged)
    elif prev_ranges and (not (framewise and (top_level or axiswise))):
        for d, dranges in dim_ranges:
            for g, drange in dranges.items():
                prange = prev_ranges.get(d, {}).get(g, None)
                if prange is None:
                    if d not in prev_ranges:
                        prev_ranges[d] = {}
                    prev_ranges[d][g] = drange
                elif g in ('factors', 'values'):
                    prev_ranges[d][g] = drange
                else:
                    prev_ranges[d][g] = util.max_range([prange, drange], combined=g == 'hard')
    else:
        ranges[group] = dict(dim_ranges)