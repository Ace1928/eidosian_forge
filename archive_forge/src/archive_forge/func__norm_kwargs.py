import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
def _norm_kwargs(self, element, ranges, opts, vdim, values=None, prefix=''):
    """
        Returns valid color normalization kwargs
        to be passed to matplotlib plot function.
        """
    dim_name = dim_range_key(vdim)
    if values is None:
        if isinstance(vdim, dim):
            values = vdim.apply(element, flat=True)
        else:
            expanded = not (isinstance(element, Dataset) and element.interface.multi and (getattr(element, 'level', None) is not None or element.interface.isunique(element, vdim.name, True)))
            values = np.asarray(element.dimension_values(vdim, expanded=expanded))
    if prefix + 'color_dim' not in self.handles:
        self.handles[prefix + 'color_dim'] = vdim
    clim = opts.pop(prefix + 'clims', None)
    if clim is None and self.clim is not None and any((util.isfinite(cl) for cl in self.clim)):
        clim = self.clim
    if clim is None:
        if not len(values):
            clim = (0, 0)
            categorical = False
        elif values.dtype.kind in 'uif':
            if dim_name in ranges:
                if self.clim_percentile and 'robust' in ranges[dim_name]:
                    clim = ranges[dim_name]['robust']
                else:
                    clim = ranges[dim_name]['combined']
            elif isinstance(vdim, dim):
                if values.dtype.kind == 'M':
                    clim = (values.min(), values.max())
                elif len(values) == 0:
                    clim = (np.nan, np.nan)
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
                            clim = (np.nanmin(values), np.nanmax(values))
                    except Exception:
                        clim = (np.nan, np.nan)
            else:
                clim = element.range(vdim)
            if self.logz:
                if clim[0] == 0:
                    clim = (values[values != 0].min(), clim[1])
            if self.symmetric:
                clim = (-np.abs(clim).max(), np.abs(clim).max())
            categorical = False
        else:
            range_key = dim_range_key(vdim)
            if range_key in ranges and 'factors' in ranges[range_key]:
                factors = ranges[range_key]['factors']
            else:
                factors = util.unique_array(values)
            clim = (0, len(factors) - 1)
            categorical = True
    else:
        categorical = values.dtype.kind not in 'uif'
    if self.cnorm == 'eq_hist':
        opts[prefix + 'norm'] = EqHistNormalize(vmin=clim[0], vmax=clim[1], rescale_discrete_levels=self.rescale_discrete_levels)
    if self.cnorm == 'log' or self.logz:
        if self.symmetric:
            norm = mpl_colors.SymLogNorm(vmin=clim[0], vmax=clim[1], linthresh=clim[1] / np.e)
        else:
            norm = mpl_colors.LogNorm(vmin=clim[0], vmax=clim[1])
        opts[prefix + 'norm'] = norm
    opts[prefix + 'vmin'] = clim[0]
    opts[prefix + 'vmax'] = clim[1]
    cmap = opts.get(prefix + 'cmap', opts.get('cmap', 'viridis'))
    if values.dtype.kind not in 'OSUM':
        ncolors = None
        if isinstance(self.color_levels, int):
            ncolors = self.color_levels
        elif isinstance(self.color_levels, list):
            ncolors = len(self.color_levels) - 1
            if isinstance(cmap, list) and len(cmap) != ncolors:
                raise ValueError('The number of colors in the colormap must match the intervals defined in the color_levels, expected %d colors found %d.' % (ncolors, len(cmap)))
        try:
            el_min, el_max = (np.nanmin(values), np.nanmax(values))
        except ValueError:
            el_min, el_max = (-np.inf, np.inf)
    else:
        ncolors = clim[-1] + 1
        el_min, el_max = (-np.inf, np.inf)
    vmin = -np.inf if opts[prefix + 'vmin'] is None else opts[prefix + 'vmin']
    vmax = np.inf if opts[prefix + 'vmax'] is None else opts[prefix + 'vmax']
    if self.cbar_extend is None:
        if el_min < vmin and el_max > vmax:
            self.cbar_extend = 'both'
        elif el_min < vmin:
            self.cbar_extend = 'min'
        elif el_max > vmax:
            self.cbar_extend = 'max'
        else:
            self.cbar_extend = 'neither'
    colors = {}
    for k, val in self.clipping_colors.items():
        if val == 'transparent':
            colors[k] = {'color': 'w', 'alpha': 0}
        elif isinstance(val, tuple):
            colors[k] = {'color': val[:3], 'alpha': val[3] if len(val) > 3 else 1}
        elif isinstance(val, str):
            color = val
            alpha = 1
            if color.startswith('#') and len(color) == 9:
                alpha = int(color[-2:], 16) / 255.0
                color = color[:-2]
            colors[k] = {'color': color, 'alpha': alpha}
    if not isinstance(cmap, mpl_colors.Colormap):
        if isinstance(cmap, dict):
            range_key = dim_range_key(vdim)
            if range_key in ranges and 'factors' in ranges[range_key]:
                factors = ranges[range_key]['factors']
            else:
                factors = util.unique_array(values)
            palette = [cmap.get(f, colors.get('NaN', {'color': self._default_nan})['color']) for f in factors]
        else:
            palette = process_cmap(cmap, ncolors, categorical=categorical)
            if isinstance(self.color_levels, list):
                palette, (vmin, vmax) = color_intervals(palette, self.color_levels, clip=(vmin, vmax))
        cmap = mpl_colors.ListedColormap(palette)
    cmap = copy.copy(cmap)
    if 'max' in colors:
        cmap.set_over(**colors['max'])
    if 'min' in colors:
        cmap.set_under(**colors['min'])
    if 'NaN' in colors:
        cmap.set_bad(**colors['NaN'])
    opts[prefix + 'cmap'] = cmap