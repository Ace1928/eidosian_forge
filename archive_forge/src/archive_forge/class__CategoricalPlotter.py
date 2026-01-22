from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
class _CategoricalPlotter(VectorPlotter):
    wide_structure = {'x': '@columns', 'y': '@values', 'hue': '@columns'}
    flat_structure = {'y': '@values'}
    _legend_attributes = ['color']

    def __init__(self, data=None, variables={}, order=None, orient=None, require_numeric=False, color=None, legend='auto'):
        super().__init__(data=data, variables=variables)
        if self.input_format == 'wide' and orient in ['h', 'y']:
            self.plot_data = self.plot_data.rename(columns={'x': 'y', 'y': 'x'})
            orig_variables = set(self.variables)
            orig_x = self.variables.pop('x', None)
            orig_y = self.variables.pop('y', None)
            orig_x_type = self.var_types.pop('x', None)
            orig_y_type = self.var_types.pop('y', None)
            if 'x' in orig_variables:
                self.variables['y'] = orig_x
                self.var_types['y'] = orig_x_type
            if 'y' in orig_variables:
                self.variables['x'] = orig_y
                self.var_types['x'] = orig_y_type
        if self.input_format == 'wide' and 'hue' in self.variables and (color is not None):
            self.plot_data.drop('hue', axis=1)
            self.variables.pop('hue')
        self.orient = infer_orient(x=self.plot_data.get('x', None), y=self.plot_data.get('y', None), orient=orient, require_numeric=False)
        self.legend = legend
        if not self.has_xy_data:
            return
        if self.orient not in self.variables:
            self.variables[self.orient] = None
            self.var_types[self.orient] = 'categorical'
            self.plot_data[self.orient] = ''
        cat_levels = categorical_order(self.plot_data[self.orient], order)
        self.var_levels[self.orient] = cat_levels

    def _hue_backcompat(self, color, palette, hue_order, force_hue=False):
        """Implement backwards compatibility for hue parametrization.

        Note: the force_hue parameter is used so that functions can be shown to
        pass existing tests during refactoring and then tested for new behavior.
        It can be removed after completion of the work.

        """
        default_behavior = color is None or palette is not None
        if force_hue and 'hue' not in self.variables and default_behavior:
            self._redundant_hue = True
            self.plot_data['hue'] = self.plot_data[self.orient]
            self.variables['hue'] = self.variables[self.orient]
            self.var_types['hue'] = 'categorical'
            hue_order = self.var_levels[self.orient]
            if isinstance(palette, dict):
                palette = {str(k): v for k, v in palette.items()}
        else:
            if 'hue' in self.variables:
                redundant = (self.plot_data['hue'] == self.plot_data[self.orient]).all()
            else:
                redundant = False
            self._redundant_hue = redundant
        if 'hue' in self.variables and palette is None and (color is not None):
            if not isinstance(color, str):
                color = mpl.colors.to_hex(color)
            palette = f'dark:{color}'
            msg = f"\n\nSetting a gradient palette using color= is deprecated and will be removed in v0.14.0. Set `palette='{palette}'` for the same effect.\n"
            warnings.warn(msg, FutureWarning, stacklevel=3)
        return (palette, hue_order)

    def _palette_without_hue_backcompat(self, palette, hue_order):
        """Provide one cycle where palette= implies hue= when not provided"""
        if 'hue' not in self.variables and palette is not None:
            msg = f'\n\nPassing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `{self.orient}` variable to `hue` and set `legend=False` for the same effect.\n'
            warnings.warn(msg, FutureWarning, stacklevel=3)
            self.legend = False
            self.plot_data['hue'] = self.plot_data[self.orient]
            self.variables['hue'] = self.variables.get(self.orient)
            self.var_types['hue'] = self.var_types.get(self.orient)
            hue_order = self.var_levels.get(self.orient)
            self._var_levels.pop('hue', None)
        return hue_order

    def _point_kwargs_backcompat(self, scale, join, kwargs):
        """Provide two cycles where scale= and join= work, but redirect to kwargs."""
        if scale is not deprecated:
            lw = mpl.rcParams['lines.linewidth'] * 1.8 * scale
            mew = lw * 0.75
            ms = lw * 2
            msg = '\n\nThe `scale` parameter is deprecated and will be removed in v0.15.0. You can now control the size of each plot element using matplotlib `Line2D` parameters (e.g., `linewidth`, `markersize`, etc.).\n'
            warnings.warn(msg, stacklevel=3)
            kwargs.update(linewidth=lw, markeredgewidth=mew, markersize=ms)
        if join is not deprecated:
            msg = '\n\nThe `join` parameter is deprecated and will be removed in v0.15.0.'
            if not join:
                msg += " You can remove the line between points with `linestyle='none'`."
                kwargs.update(linestyle='')
            msg += '\n'
            warnings.warn(msg, stacklevel=3)

    def _err_kws_backcompat(self, err_kws, errcolor, errwidth, capsize):
        """Provide two cycles where existing signature-level err_kws are handled."""

        def deprecate_err_param(name, key, val):
            if val is deprecated:
                return
            suggest = f"err_kws={{'{key}': {val!r}}}"
            msg = f'\n\nThe `{name}` parameter is deprecated. And will be removed in v0.15.0. Pass `{suggest}` instead.\n'
            warnings.warn(msg, FutureWarning, stacklevel=4)
            err_kws[key] = val
        if errcolor is not None:
            deprecate_err_param('errcolor', 'color', errcolor)
        deprecate_err_param('errwidth', 'linewidth', errwidth)
        if capsize is None:
            capsize = 0
            msg = '\n\nPassing `capsize=None` is deprecated and will be removed in v0.15.0. Pass `capsize=0` to disable caps.\n'
            warnings.warn(msg, FutureWarning, stacklevel=3)
        return (err_kws, capsize)

    def _violin_scale_backcompat(self, scale, scale_hue, density_norm, common_norm):
        """Provide two cycles of backcompat for scale kwargs"""
        if scale is not deprecated:
            density_norm = scale
            msg = f'\n\nThe `scale` parameter has been renamed and will be removed in v0.15.0. Pass `density_norm={scale!r}` for the same effect.'
            warnings.warn(msg, FutureWarning, stacklevel=3)
        if scale_hue is not deprecated:
            common_norm = scale_hue
            msg = f'\n\nThe `scale_hue` parameter has been replaced and will be removed in v0.15.0. Pass `common_norm={not scale_hue}` for the same effect.'
            warnings.warn(msg, FutureWarning, stacklevel=3)
        return (density_norm, common_norm)

    def _violin_bw_backcompat(self, bw, bw_method):
        """Provide two cycles of backcompat for violin bandwidth parameterization."""
        if bw is not deprecated:
            bw_method = bw
            msg = dedent(f'\n\n                The `bw` parameter is deprecated in favor of `bw_method`/`bw_adjust`.\n                Setting `bw_method={bw!r}`, but please see docs for the new parameters\n                and update your code. This will become an error in seaborn v0.15.0.\n            ')
            warnings.warn(msg, FutureWarning, stacklevel=3)
        return bw_method

    def _boxen_scale_backcompat(self, scale, width_method):
        """Provide two cycles of backcompat for scale kwargs"""
        if scale is not deprecated:
            width_method = scale
            msg = f'\n\nThe `scale` parameter has been renamed to `width_method` and will be removed in v0.15. Pass `width_method={scale!r}'
            if scale == 'area':
                msg += ", but note that the result for 'area' will appear different."
            else:
                msg += ' for the same effect.'
            warnings.warn(msg, FutureWarning, stacklevel=3)
        return width_method

    def _complement_color(self, color, base_color, hue_map):
        """Allow a color to be set automatically using a basis of comparison."""
        if color == 'gray':
            msg = 'Use "auto" to set automatic grayscale colors. From v0.14.0, "gray" will default to matplotlib\'s definition.'
            warnings.warn(msg, FutureWarning, stacklevel=3)
            color = 'auto'
        elif color is None or color is default:
            color = 'auto'
        if color != 'auto':
            return color
        if hue_map.lookup_table is None:
            if base_color is None:
                return None
            basis = [mpl.colors.to_rgb(base_color)]
        else:
            basis = [mpl.colors.to_rgb(c) for c in hue_map.lookup_table.values()]
        unique_colors = np.unique(basis, axis=0)
        light_vals = [rgb_to_hls(*rgb[:3])[1] for rgb in unique_colors]
        lum = min(light_vals) * 0.6
        return (lum, lum, lum)

    def _map_prop_with_hue(self, name, value, fallback, plot_kws):
        """Support pointplot behavior of modifying the marker/linestyle with hue."""
        if value is default:
            value = plot_kws.pop(name, fallback)
        if 'hue' in self.variables:
            levels = self._hue_map.levels
            if isinstance(value, list):
                mapping = {k: v for k, v in zip(levels, value)}
            else:
                mapping = {k: value for k in levels}
        else:
            mapping = {None: value}
        return mapping

    def _adjust_cat_axis(self, ax, axis):
        """Set ticks and limits for a categorical variable."""
        if self.var_types[axis] != 'categorical':
            return
        if self.plot_data[axis].empty:
            return
        n = len(getattr(ax, f'get_{axis}ticks')())
        if axis == 'x':
            ax.xaxis.grid(False)
            ax.set_xlim(-0.5, n - 0.5, auto=None)
        else:
            ax.yaxis.grid(False)
            ax.set_ylim(n - 0.5, -0.5, auto=None)

    def _dodge_needed(self):
        """Return True when use of `hue` would cause overlaps."""
        groupers = list({self.orient, 'col', 'row'} & set(self.variables))
        if 'hue' in self.variables:
            orient = self.plot_data[groupers].value_counts()
            paired = self.plot_data[[*groupers, 'hue']].value_counts()
            return orient.size != paired.size
        return False

    def _dodge(self, keys, data):
        """Apply a dodge transform to coordinates in place."""
        if 'hue' not in self.variables:
            return
        hue_idx = self._hue_map.levels.index(keys['hue'])
        n = len(self._hue_map.levels)
        data['width'] /= n
        full_width = data['width'] * n
        offset = data['width'] * hue_idx + data['width'] / 2 - full_width / 2
        data[self.orient] += offset

    def _invert_scale(self, ax, data, vars=('x', 'y')):
        """Undo scaling after computation so data are plotted correctly."""
        for var in vars:
            _, inv = _get_transform_functions(ax, var[0])
            if var == self.orient and 'width' in data:
                hw = data['width'] / 2
                data['edge'] = inv(data[var] - hw)
                data['width'] = inv(data[var] + hw) - data['edge'].to_numpy()
            for suf in ['', 'min', 'max']:
                if (col := f'{var}{suf}') in data:
                    data[col] = inv(data[col])

    def _configure_legend(self, ax, func, common_kws=None, semantic_kws=None):
        if self.legend == 'auto':
            show_legend = not self._redundant_hue and self.input_format != 'wide'
        else:
            show_legend = bool(self.legend)
        if show_legend:
            self.add_legend_data(ax, func, common_kws, semantic_kws=semantic_kws)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(title=self.legend_title)

    @property
    def _native_width(self):
        """Return unit of width separating categories on native numeric scale."""
        if self.var_types[self.orient] == 'categorical':
            return 1
        unique_values = np.unique(self.comp_data[self.orient])
        if len(unique_values) > 1:
            native_width = np.nanmin(np.diff(unique_values))
        else:
            native_width = 1
        return native_width

    def _nested_offsets(self, width, dodge):
        """Return offsets for each hue level for dodged plots."""
        offsets = None
        if 'hue' in self.variables and self._hue_map.levels is not None:
            n_levels = len(self._hue_map.levels)
            if dodge:
                each_width = width / n_levels
                offsets = np.linspace(0, width - each_width, n_levels)
                offsets -= offsets.mean()
            else:
                offsets = np.zeros(n_levels)
        return offsets

    def plot_strips(self, jitter, dodge, color, plot_kws):
        width = 0.8 * self._native_width
        offsets = self._nested_offsets(width, dodge)
        if jitter is True:
            jlim = 0.1
        else:
            jlim = float(jitter)
        if 'hue' in self.variables and dodge and (self._hue_map.levels is not None):
            jlim /= len(self._hue_map.levels)
        jlim *= self._native_width
        jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)
        iter_vars = [self.orient]
        if dodge:
            iter_vars.append('hue')
        ax = self.ax
        dodge_move = jitter_move = 0
        if 'marker' in plot_kws and (not MarkerStyle(plot_kws['marker']).is_filled()):
            plot_kws.pop('edgecolor', None)
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
            ax = self._get_axes(sub_vars)
            if offsets is not None and (offsets != 0).any():
                dodge_move = offsets[sub_data['hue'].map(self._hue_map.levels.index)]
            jitter_move = jitterer(size=len(sub_data)) if len(sub_data) > 1 else 0
            adjusted_data = sub_data[self.orient] + dodge_move + jitter_move
            sub_data[self.orient] = adjusted_data
            self._invert_scale(ax, sub_data)
            points = ax.scatter(sub_data['x'], sub_data['y'], color=color, **plot_kws)
            if 'hue' in self.variables:
                points.set_facecolors(self._hue_map(sub_data['hue']))
        self._configure_legend(ax, _scatter_legend_artist, common_kws=plot_kws)

    def plot_swarms(self, dodge, color, warn_thresh, plot_kws):
        width = 0.8 * self._native_width
        offsets = self._nested_offsets(width, dodge)
        iter_vars = [self.orient]
        if dodge:
            iter_vars.append('hue')
        ax = self.ax
        point_collections = {}
        dodge_move = 0
        if 'marker' in plot_kws and (not MarkerStyle(plot_kws['marker']).is_filled()):
            plot_kws.pop('edgecolor', None)
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
            ax = self._get_axes(sub_vars)
            if offsets is not None:
                dodge_move = offsets[sub_data['hue'].map(self._hue_map.levels.index)]
            if not sub_data.empty:
                sub_data[self.orient] = sub_data[self.orient] + dodge_move
            self._invert_scale(ax, sub_data)
            points = ax.scatter(sub_data['x'], sub_data['y'], color=color, **plot_kws)
            if 'hue' in self.variables:
                points.set_facecolors(self._hue_map(sub_data['hue']))
            if not sub_data.empty:
                point_collections[ax, sub_data[self.orient].iloc[0]] = points
        beeswarm = Beeswarm(width=width, orient=self.orient, warn_thresh=warn_thresh)
        for (ax, center), points in point_collections.items():
            if points.get_offsets().shape[0] > 1:

                def draw(points, renderer, *, center=center):
                    beeswarm(points, center)
                    if self.orient == 'y':
                        scalex = False
                        scaley = ax.get_autoscaley_on()
                    else:
                        scalex = ax.get_autoscalex_on()
                        scaley = False
                    fixed_scale = self.var_types[self.orient] == 'categorical'
                    ax.update_datalim(points.get_datalim(ax.transData))
                    if not fixed_scale and (scalex or scaley):
                        ax.autoscale_view(scalex=scalex, scaley=scaley)
                    super(points.__class__, points).draw(renderer)
                points.draw = draw.__get__(points)
        _draw_figure(ax.figure)
        self._configure_legend(ax, _scatter_legend_artist, plot_kws)

    def plot_boxes(self, width, dodge, gap, fill, whis, color, linecolor, linewidth, fliersize, plot_kws):
        iter_vars = ['hue']
        value_var = {'x': 'y', 'y': 'x'}[self.orient]

        def get_props(element, artist=mpl.lines.Line2D):
            return normalize_kwargs(plot_kws.pop(f'{element}props', {}), artist)
        if not fill and linewidth is None:
            linewidth = mpl.rcParams['lines.linewidth']
        bootstrap = plot_kws.pop('bootstrap', mpl.rcParams['boxplot.bootstrap'])
        plot_kws.setdefault('shownotches', plot_kws.pop('notch', False))
        box_artist = mpl.patches.Rectangle if fill else mpl.lines.Line2D
        props = {'box': get_props('box', box_artist), 'median': get_props('median'), 'whisker': get_props('whisker'), 'flier': get_props('flier'), 'cap': get_props('cap')}
        props['median'].setdefault('solid_capstyle', 'butt')
        props['whisker'].setdefault('solid_capstyle', 'butt')
        props['flier'].setdefault('markersize', fliersize)
        ax = self.ax
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
            ax = self._get_axes(sub_vars)
            grouped = sub_data.groupby(self.orient)[value_var]
            positions = sorted(sub_data[self.orient].unique().astype(float))
            value_data = [x.to_numpy() for _, x in grouped]
            stats = pd.DataFrame(mpl.cbook.boxplot_stats(value_data, whis=whis, bootstrap=bootstrap))
            orig_width = width * self._native_width
            data = pd.DataFrame({self.orient: positions, 'width': orig_width})
            if dodge:
                self._dodge(sub_vars, data)
            if gap:
                data['width'] *= 1 - gap
            capwidth = plot_kws.get('capwidths', 0.5 * data['width'])
            self._invert_scale(ax, data)
            _, inv = _get_transform_functions(ax, value_var)
            for stat in ['mean', 'med', 'q1', 'q3', 'cilo', 'cihi', 'whislo', 'whishi']:
                stats[stat] = inv(stats[stat])
            stats['fliers'] = stats['fliers'].map(inv)
            linear_orient_scale = getattr(ax, f'get_{self.orient}scale')() == 'linear'
            maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
            if fill:
                boxprops = {'facecolor': maincolor, 'edgecolor': linecolor, **props['box']}
                medianprops = {'color': linecolor, **props['median']}
                whiskerprops = {'color': linecolor, **props['whisker']}
                flierprops = {'markeredgecolor': linecolor, **props['flier']}
                capprops = {'color': linecolor, **props['cap']}
            else:
                boxprops = {'color': maincolor, **props['box']}
                medianprops = {'color': maincolor, **props['median']}
                whiskerprops = {'color': maincolor, **props['whisker']}
                flierprops = {'markeredgecolor': maincolor, **props['flier']}
                capprops = {'color': maincolor, **props['cap']}
            if linewidth is not None:
                for prop_dict in [boxprops, medianprops, whiskerprops, capprops]:
                    prop_dict.setdefault('linewidth', linewidth)
            default_kws = dict(bxpstats=stats.to_dict('records'), positions=data[self.orient], widths=data['width'] if linear_orient_scale else 0, patch_artist=fill, vert=self.orient == 'x', manage_ticks=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, **{} if _version_predates(mpl, '3.6.0') else {'capwidths': capwidth})
            boxplot_kws = {**default_kws, **plot_kws}
            artists = ax.bxp(**boxplot_kws)
            ori_idx = ['x', 'y'].index(self.orient)
            if not linear_orient_scale:
                for i, box in enumerate(data.to_dict('records')):
                    p0 = box['edge']
                    p1 = box['edge'] + box['width']
                    if artists['boxes']:
                        box_artist = artists['boxes'][i]
                        if fill:
                            box_verts = box_artist.get_path().vertices.T
                        else:
                            box_verts = box_artist.get_data()
                        box_verts[ori_idx][0] = p0
                        box_verts[ori_idx][3:] = p0
                        box_verts[ori_idx][1:3] = p1
                        if not fill:
                            box_artist.set_data(box_verts)
                        ax.update_datalim(np.transpose(box_verts), updatex=self.orient == 'x', updatey=self.orient == 'y')
                    if artists['medians']:
                        verts = artists['medians'][i].get_xydata().T
                        verts[ori_idx][:] = (p0, p1)
                        artists['medians'][i].set_data(verts)
                    if artists['caps']:
                        f_fwd, f_inv = _get_transform_functions(ax, self.orient)
                        for line in artists['caps'][2 * i:2 * i + 2]:
                            p0 = f_inv(f_fwd(box[self.orient]) - capwidth[i] / 2)
                            p1 = f_inv(f_fwd(box[self.orient]) + capwidth[i] / 2)
                            verts = line.get_xydata().T
                            verts[ori_idx][:] = (p0, p1)
                            line.set_data(verts)
            ax.add_container(BoxPlotContainer(artists))
        legend_artist = _get_patch_legend_artist(fill)
        self._configure_legend(ax, legend_artist, boxprops)

    def plot_boxens(self, width, dodge, gap, fill, color, linecolor, linewidth, width_method, k_depth, outlier_prop, trust_alpha, showfliers, box_kws, flier_kws, line_kws, plot_kws):
        iter_vars = [self.orient, 'hue']
        value_var = {'x': 'y', 'y': 'x'}[self.orient]
        estimator = LetterValues(k_depth, outlier_prop, trust_alpha)
        width_method_options = ['exponential', 'linear', 'area']
        _check_argument('width_method', width_method_options, width_method)
        box_kws = plot_kws if box_kws is None else {**plot_kws, **box_kws}
        flier_kws = {} if flier_kws is None else flier_kws.copy()
        line_kws = {} if line_kws is None else line_kws.copy()
        if linewidth is None:
            if fill:
                linewidth = 0.5 * mpl.rcParams['lines.linewidth']
            else:
                linewidth = mpl.rcParams['lines.linewidth']
        ax = self.ax
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
            ax = self._get_axes(sub_vars)
            _, inv_ori = _get_transform_functions(ax, self.orient)
            _, inv_val = _get_transform_functions(ax, value_var)
            lv_data = estimator(sub_data[value_var])
            n = lv_data['k'] * 2 - 1
            vals = lv_data['values']
            pos_data = pd.DataFrame({self.orient: [sub_vars[self.orient]], 'width': [width * self._native_width]})
            if dodge:
                self._dodge(sub_vars, pos_data)
            if gap:
                pos_data['width'] *= 1 - gap
            levels = lv_data['levels']
            exponent = (levels - 1 - lv_data['k']).astype(float)
            if width_method == 'linear':
                rel_widths = levels + 1
            elif width_method == 'exponential':
                rel_widths = 2 ** exponent
            elif width_method == 'area':
                tails = levels < lv_data['k'] - 1
                rel_widths = 2 ** (exponent - tails) / np.diff(lv_data['values'])
            center = pos_data[self.orient].item()
            widths = rel_widths / rel_widths.max() * pos_data['width'].item()
            box_vals = inv_val(vals)
            box_pos = inv_ori(center - widths / 2)
            box_heights = inv_val(vals[1:]) - inv_val(vals[:-1])
            box_widths = inv_ori(center + widths / 2) - inv_ori(center - widths / 2)
            maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
            flier_colors = {'facecolor': 'none', 'edgecolor': '.45' if fill else maincolor}
            if fill:
                cmap = light_palette(maincolor, as_cmap=True)
                boxcolors = cmap(2 ** ((exponent + 2) / 3))
            else:
                boxcolors = maincolor
            boxen = []
            for i in range(n):
                if self.orient == 'x':
                    xy = (box_pos[i], box_vals[i])
                    w, h = (box_widths[i], box_heights[i])
                else:
                    xy = (box_vals[i], box_pos[i])
                    w, h = (box_heights[i], box_widths[i])
                boxen.append(Rectangle(xy, w, h))
            if fill:
                box_colors = {'facecolors': boxcolors, 'edgecolors': linecolor}
            else:
                box_colors = {'facecolors': 'none', 'edgecolors': boxcolors}
            collection_kws = {**box_colors, 'linewidth': linewidth, **box_kws}
            ax.add_collection(PatchCollection(boxen, **collection_kws), autolim=False)
            ax.update_datalim(np.column_stack([box_vals, box_vals]), updatex=self.orient == 'y', updatey=self.orient == 'x')
            med = lv_data['median']
            hw = pos_data['width'].item() / 2
            if self.orient == 'x':
                x, y = (inv_ori([center - hw, center + hw]), inv_val([med, med]))
            else:
                x, y = (inv_val([med, med]), inv_ori([center - hw, center + hw]))
            default_kws = {'color': linecolor if fill else maincolor, 'solid_capstyle': 'butt', 'linewidth': 1.25 * linewidth}
            ax.plot(x, y, **{**default_kws, **line_kws})
            if showfliers:
                vals = inv_val(lv_data['fliers'])
                pos = np.full(len(vals), inv_ori(pos_data[self.orient].item()))
                x, y = (pos, vals) if self.orient == 'x' else (vals, pos)
                ax.scatter(x, y, **{**flier_colors, 's': 25, **flier_kws})
        ax.autoscale_view(scalex=self.orient == 'y', scaley=self.orient == 'x')
        legend_artist = _get_patch_legend_artist(fill)
        common_kws = {**box_kws, 'linewidth': linewidth, 'edgecolor': linecolor}
        self._configure_legend(ax, legend_artist, common_kws)

    def plot_violins(self, width, dodge, gap, split, color, fill, linecolor, linewidth, inner, density_norm, common_norm, kde_kws, inner_kws, plot_kws):
        iter_vars = [self.orient, 'hue']
        value_var = {'x': 'y', 'y': 'x'}[self.orient]
        inner_options = ['box', 'quart', 'stick', 'point', None]
        _check_argument('inner', inner_options, inner, prefix=True)
        _check_argument('density_norm', ['area', 'count', 'width'], density_norm)
        if linewidth is None:
            if fill:
                linewidth = 1.25 * mpl.rcParams['patch.linewidth']
            else:
                linewidth = mpl.rcParams['lines.linewidth']
        if inner is not None and inner.startswith('box'):
            box_width = inner_kws.pop('box_width', linewidth * 4.5)
            whis_width = inner_kws.pop('whis_width', box_width / 3)
            marker = inner_kws.pop('marker', '_' if self.orient == 'x' else '|')
        kde = KDE(**kde_kws)
        ax = self.ax
        violin_data = []
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
            sub_data['weight'] = sub_data.get('weights', 1)
            stat_data = kde._transform(sub_data, value_var, [])
            maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
            if not fill:
                linecolor = maincolor
                maincolor = 'none'
            default_kws = dict(facecolor=maincolor, edgecolor=linecolor, linewidth=linewidth)
            violin_data.append({'position': sub_vars[self.orient], 'observations': sub_data[value_var], 'density': stat_data['density'], 'support': stat_data[value_var], 'kwargs': {**default_kws, **plot_kws}, 'sub_vars': sub_vars, 'ax': self._get_axes(sub_vars)})

        def vars_to_key(sub_vars):
            return tuple(((k, v) for k, v in sub_vars.items() if k != self.orient))
        norm_keys = [vars_to_key(violin['sub_vars']) for violin in violin_data]
        if common_norm:
            common_max_density = np.nanmax([v['density'].max() for v in violin_data])
            common_max_count = np.nanmax([len(v['observations']) for v in violin_data])
            max_density = {key: common_max_density for key in norm_keys}
            max_count = {key: common_max_count for key in norm_keys}
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
                max_density = {key: np.nanmax([v['density'].max() for v in violin_data if vars_to_key(v['sub_vars']) == key]) for key in norm_keys}
            max_count = {key: np.nanmax([len(v['observations']) for v in violin_data if vars_to_key(v['sub_vars']) == key]) for key in norm_keys}
        real_width = width * self._native_width
        for violin in violin_data:
            index = pd.RangeIndex(0, max(len(violin['support']), 1))
            data = pd.DataFrame({self.orient: violin['position'], value_var: violin['support'], 'density': violin['density'], 'width': real_width}, index=index)
            if dodge:
                self._dodge(violin['sub_vars'], data)
            if gap:
                data['width'] *= 1 - gap
            norm_key = vars_to_key(violin['sub_vars'])
            hw = data['width'] / 2
            peak_density = violin['density'].max()
            if np.isnan(peak_density):
                span = 1
            elif density_norm == 'area':
                span = data['density'] / max_density[norm_key]
            elif density_norm == 'count':
                count = len(violin['observations'])
                span = data['density'] / peak_density * (count / max_count[norm_key])
            elif density_norm == 'width':
                span = data['density'] / peak_density
            span = span * hw * (2 if split else 1)
            right_side = 0 if 'hue' not in self.variables else self._hue_map.levels.index(violin['sub_vars']['hue']) % 2
            if split:
                offsets = (hw, span - hw) if right_side else (span - hw, hw)
            else:
                offsets = (span, span)
            ax = violin['ax']
            _, invx = _get_transform_functions(ax, 'x')
            _, invy = _get_transform_functions(ax, 'y')
            inv_pos = {'x': invx, 'y': invy}[self.orient]
            inv_val = {'x': invx, 'y': invy}[value_var]
            linecolor = violin['kwargs']['edgecolor']
            if np.isnan(peak_density):
                pos = data[self.orient].iloc[0]
                val = violin['observations'].mean()
                if self.orient == 'x':
                    x, y = ([pos - offsets[0], pos + offsets[1]], [val, val])
                else:
                    x, y = ([val, val], [pos - offsets[0], pos + offsets[1]])
                ax.plot(invx(x), invy(y), color=linecolor, linewidth=linewidth)
                continue
            plot_func = {'x': ax.fill_betweenx, 'y': ax.fill_between}[self.orient]
            plot_func(inv_val(data[value_var]), inv_pos(data[self.orient] - offsets[0]), inv_pos(data[self.orient] + offsets[1]), **violin['kwargs'])
            obs = violin['observations']
            pos_dict = {self.orient: violin['position'], 'width': real_width}
            if dodge:
                self._dodge(violin['sub_vars'], pos_dict)
            if gap:
                pos_dict['width'] *= 1 - gap
            if inner is None:
                continue
            elif inner.startswith('point'):
                pos = np.array([pos_dict[self.orient]] * len(obs))
                if split:
                    pos += (-1 if right_side else 1) * pos_dict['width'] / 2
                x, y = (pos, obs) if self.orient == 'x' else (obs, pos)
                kws = {'color': linecolor, 'edgecolor': linecolor, 's': (linewidth * 2) ** 2, 'zorder': violin['kwargs'].get('zorder', 2) + 1, **inner_kws}
                ax.scatter(invx(x), invy(y), **kws)
            elif inner.startswith('stick'):
                pos0 = np.interp(obs, data[value_var], data[self.orient] - offsets[0])
                pos1 = np.interp(obs, data[value_var], data[self.orient] + offsets[1])
                pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
                val_pts = np.stack([inv_val(obs), inv_val(obs)])
                segments = np.stack([pos_pts, val_pts]).transpose(2, 1, 0)
                if self.orient == 'y':
                    segments = segments[:, :, ::-1]
                kws = {'color': linecolor, 'linewidth': linewidth / 2, **inner_kws}
                lines = mpl.collections.LineCollection(segments, **kws)
                ax.add_collection(lines, autolim=False)
            elif inner.startswith('quart'):
                stats = np.percentile(obs, [25, 50, 75])
                pos0 = np.interp(stats, data[value_var], data[self.orient] - offsets[0])
                pos1 = np.interp(stats, data[value_var], data[self.orient] + offsets[1])
                pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
                val_pts = np.stack([inv_val(stats), inv_val(stats)])
                segments = np.stack([pos_pts, val_pts]).transpose(2, 0, 1)
                if self.orient == 'y':
                    segments = segments[:, ::-1, :]
                dashes = [(1.25, 0.75), (2.5, 1), (1.25, 0.75)]
                for i, segment in enumerate(segments):
                    kws = {'color': linecolor, 'linewidth': linewidth, 'dashes': dashes[i], **inner_kws}
                    ax.plot(*segment, **kws)
            elif inner.startswith('box'):
                stats = mpl.cbook.boxplot_stats(obs)[0]
                pos = np.array(pos_dict[self.orient])
                if split:
                    pos += (-1 if right_side else 1) * pos_dict['width'] / 2
                pos = ([pos, pos], [pos, pos], [pos])
                val = ([stats['whislo'], stats['whishi']], [stats['q1'], stats['q3']], [stats['med']])
                if self.orient == 'x':
                    (x0, x1, x2), (y0, y1, y2) = (pos, val)
                else:
                    (x0, x1, x2), (y0, y1, y2) = (val, pos)
                if split:
                    offset = (1 if right_side else -1) * box_width / 72 / 2
                    dx, dy = (offset, 0) if self.orient == 'x' else (0, -offset)
                    trans = ax.transData + mpl.transforms.ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
                else:
                    trans = ax.transData
                line_kws = {'color': linecolor, 'transform': trans, **inner_kws, 'linewidth': whis_width}
                ax.plot(invx(x0), invy(y0), **line_kws)
                line_kws['linewidth'] = box_width
                ax.plot(invx(x1), invy(y1), **line_kws)
                dot_kws = {'marker': marker, 'markersize': box_width / 1.2, 'markeredgewidth': box_width / 5, 'transform': trans, **inner_kws, 'markeredgecolor': 'w', 'markerfacecolor': 'w', 'color': linecolor}
                ax.plot(invx(x2), invy(y2), **dot_kws)
        legend_artist = _get_patch_legend_artist(fill)
        common_kws = {**plot_kws, 'linewidth': linewidth, 'edgecolor': linecolor}
        self._configure_legend(ax, legend_artist, common_kws)

    def plot_points(self, aggregator, markers, linestyles, dodge, color, capsize, err_kws, plot_kws):
        agg_var = {'x': 'y', 'y': 'x'}[self.orient]
        iter_vars = ['hue']
        plot_kws = normalize_kwargs(plot_kws, mpl.lines.Line2D)
        plot_kws.setdefault('linewidth', mpl.rcParams['lines.linewidth'] * 1.8)
        plot_kws.setdefault('markeredgewidth', plot_kws['linewidth'] * 0.75)
        plot_kws.setdefault('markersize', plot_kws['linewidth'] * np.sqrt(2 * np.pi))
        markers = self._map_prop_with_hue('marker', markers, 'o', plot_kws)
        linestyles = self._map_prop_with_hue('linestyle', linestyles, '-', plot_kws)
        base_positions = self.var_levels[self.orient]
        if self.var_types[self.orient] == 'categorical':
            min_cat_val = int(self.comp_data[self.orient].min())
            max_cat_val = int(self.comp_data[self.orient].max())
            base_positions = [i for i in range(min_cat_val, max_cat_val + 1)]
        n_hue_levels = 0 if self._hue_map.levels is None else len(self._hue_map.levels)
        if dodge is True:
            dodge = 0.025 * n_hue_levels
        ax = self.ax
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
            ax = self._get_axes(sub_vars)
            ori_axis = getattr(ax, f'{self.orient}axis')
            transform, _ = _get_transform_functions(ax, self.orient)
            positions = transform(ori_axis.convert_units(base_positions))
            agg_data = sub_data if sub_data.empty else sub_data.groupby(self.orient).apply(aggregator, agg_var, **groupby_apply_include_groups(False)).reindex(pd.Index(positions, name=self.orient)).reset_index()
            if dodge:
                hue_idx = self._hue_map.levels.index(sub_vars['hue'])
                step_size = dodge / (n_hue_levels - 1)
                offset = -dodge / 2 + step_size * hue_idx
                agg_data[self.orient] += offset * self._native_width
            self._invert_scale(ax, agg_data)
            sub_kws = plot_kws.copy()
            sub_kws.update(marker=markers[sub_vars.get('hue')], linestyle=linestyles[sub_vars.get('hue')], color=self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color)
            line, = ax.plot(agg_data['x'], agg_data['y'], **sub_kws)
            sub_err_kws = err_kws.copy()
            line_props = line.properties()
            for prop in ['color', 'linewidth', 'alpha', 'zorder']:
                sub_err_kws.setdefault(prop, line_props[prop])
            if aggregator.error_method is not None:
                self.plot_errorbars(ax, agg_data, capsize, sub_err_kws)
        legend_artist = partial(mpl.lines.Line2D, [], [])
        semantic_kws = {'hue': {'marker': markers, 'linestyle': linestyles}}
        self._configure_legend(ax, legend_artist, sub_kws, semantic_kws)

    def plot_bars(self, aggregator, dodge, gap, width, fill, color, capsize, err_kws, plot_kws):
        agg_var = {'x': 'y', 'y': 'x'}[self.orient]
        iter_vars = ['hue']
        ax = self.ax
        if self._hue_map.levels is None:
            dodge = False
        if dodge and capsize is not None:
            capsize = capsize / len(self._hue_map.levels)
        if not fill:
            plot_kws.setdefault('linewidth', 1.5 * mpl.rcParams['lines.linewidth'])
        err_kws.setdefault('linewidth', 1.5 * mpl.rcParams['lines.linewidth'])
        for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
            ax = self._get_axes(sub_vars)
            agg_data = sub_data if sub_data.empty else sub_data.groupby(self.orient).apply(aggregator, agg_var, **groupby_apply_include_groups(False)).reset_index()
            agg_data['width'] = width * self._native_width
            if dodge:
                self._dodge(sub_vars, agg_data)
            if gap:
                agg_data['width'] *= 1 - gap
            agg_data['edge'] = agg_data[self.orient] - agg_data['width'] / 2
            self._invert_scale(ax, agg_data)
            if self.orient == 'x':
                bar_func = ax.bar
                kws = dict(x=agg_data['edge'], height=agg_data['y'], width=agg_data['width'])
            else:
                bar_func = ax.barh
                kws = dict(y=agg_data['edge'], width=agg_data['x'], height=agg_data['width'])
            main_color = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
            kws['align'] = 'edge'
            if fill:
                kws.update(color=main_color, facecolor=main_color)
            else:
                kws.update(color=main_color, edgecolor=main_color, facecolor='none')
            bar_func(**{**kws, **plot_kws})
            if aggregator.error_method is not None:
                self.plot_errorbars(ax, agg_data, capsize, {'color': '.26' if fill else main_color, **err_kws})
        legend_artist = _get_patch_legend_artist(fill)
        self._configure_legend(ax, legend_artist, plot_kws)

    def plot_errorbars(self, ax, data, capsize, err_kws):
        var = {'x': 'y', 'y': 'x'}[self.orient]
        for row in data.to_dict('records'):
            row = dict(row)
            pos = np.array([row[self.orient], row[self.orient]])
            val = np.array([row[f'{var}min'], row[f'{var}max']])
            if capsize:
                cw = capsize * self._native_width / 2
                scl, inv = _get_transform_functions(ax, self.orient)
                cap = (inv(scl(pos[0]) - cw), inv(scl(pos[1]) + cw))
                pos = np.concatenate([[*cap, np.nan], pos, [np.nan, *cap]])
                val = np.concatenate([[val[0], val[0], np.nan], val, [np.nan, val[-1], val[-1]]])
            if self.orient == 'x':
                args = (pos, val)
            else:
                args = (val, pos)
            ax.plot(*args, **err_kws)