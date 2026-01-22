from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
class VectorPlotter:
    """Base class for objects underlying *plot functions."""
    wide_structure = {'x': '@index', 'y': '@values', 'hue': '@columns', 'style': '@columns'}
    flat_structure = {'x': '@index', 'y': '@values'}
    _default_size_range = (1, 2)

    def __init__(self, data=None, variables={}):
        self._var_levels = {}
        self._var_ordered = {'x': False, 'y': False}
        self.assign_variables(data, variables)
        for var in ['hue', 'size', 'style']:
            if var in variables:
                getattr(self, f'map_{var}')()

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        return bool({'x', 'y'} & set(self.variables))

    @property
    def var_levels(self):
        """Property interface to ordered list of variables levels.

        Each time it's accessed, it updates the var_levels dictionary with the
        list of levels in the current semantic mappers. But it also allows the
        dictionary to persist, so it can be used to set levels by a key. This is
        used to track the list of col/row levels using an attached FacetGrid
        object, but it's kind of messy and ideally fixed by improving the
        faceting logic so it interfaces better with the modern approach to
        tracking plot variables.

        """
        for var in self.variables:
            if (map_obj := getattr(self, f'_{var}_map', None)) is not None:
                self._var_levels[var] = map_obj.levels
        return self._var_levels

    def assign_variables(self, data=None, variables={}):
        """Define plot variables, optionally using lookup from `data`."""
        x = variables.get('x', None)
        y = variables.get('y', None)
        if x is None and y is None:
            self.input_format = 'wide'
            frame, names = self._assign_variables_wideform(data, **variables)
        else:
            self.input_format = 'long'
            plot_data = PlotData(data, variables)
            frame = plot_data.frame
            names = plot_data.names
        self.plot_data = frame
        self.variables = names
        self.var_types = {v: variable_type(frame[v], boolean_type='numeric' if v in 'xy' else 'categorical') for v in names}
        return self

    def _assign_variables_wideform(self, data=None, **kwargs):
        """Define plot variables given wide-form data.

        Parameters
        ----------
        data : flat vector or collection of vectors
            Data can be a vector or mapping that is coerceable to a Series
            or a sequence- or mapping-based collection of such vectors, or a
            rectangular numpy array, or a Pandas DataFrame.
        kwargs : variable -> data mappings
            Behavior with keyword arguments is currently undefined.

        Returns
        -------
        plot_data : :class:`pandas.DataFrame`
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        variables : dict
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        """
        assigned = [k for k, v in kwargs.items() if v is not None]
        if any(assigned):
            s = 's' if len(assigned) > 1 else ''
            err = f'The following variable{s} cannot be assigned with wide-form data: '
            err += ', '.join((f'`{v}`' for v in assigned))
            raise ValueError(err)
        empty = data is None or not len(data)
        if isinstance(data, dict):
            values = data.values()
        else:
            values = np.atleast_1d(np.asarray(data, dtype=object))
        flat = not any((isinstance(v, Iterable) and (not isinstance(v, (str, bytes))) for v in values))
        if empty:
            plot_data = pd.DataFrame()
            variables = {}
        elif flat:
            flat_data = pd.Series(data).copy()
            names = {'@values': flat_data.name, '@index': flat_data.index.name}
            plot_data = {}
            variables = {}
            for var in ['x', 'y']:
                if var in self.flat_structure:
                    attr = self.flat_structure[var]
                    plot_data[var] = getattr(flat_data, attr[1:])
                    variables[var] = names[self.flat_structure[var]]
            plot_data = pd.DataFrame(plot_data)
        else:
            if isinstance(data, Sequence):
                data_dict = {}
                for i, var in enumerate(data):
                    key = getattr(var, 'name', i)
                    data_dict[key] = pd.Series(var)
                data = data_dict
            if isinstance(data, Mapping):
                data = {key: pd.Series(val) for key, val in data.items()}
            wide_data = pd.DataFrame(data, copy=True)
            numeric_cols = [k for k, v in wide_data.items() if variable_type(v) == 'numeric']
            wide_data = wide_data[numeric_cols]
            melt_kws = {'var_name': '@columns', 'value_name': '@values'}
            use_index = '@index' in self.wide_structure.values()
            if use_index:
                melt_kws['id_vars'] = '@index'
                try:
                    orig_categories = wide_data.columns.categories
                    orig_ordered = wide_data.columns.ordered
                    wide_data.columns = wide_data.columns.add_categories('@index')
                except AttributeError:
                    category_columns = False
                else:
                    category_columns = True
                wide_data['@index'] = wide_data.index.to_series()
            plot_data = wide_data.melt(**melt_kws)
            if use_index and category_columns:
                plot_data['@columns'] = pd.Categorical(plot_data['@columns'], orig_categories, orig_ordered)
            for var, attr in self.wide_structure.items():
                plot_data[var] = plot_data[attr]
            variables = {}
            for var, attr in self.wide_structure.items():
                obj = getattr(wide_data, attr[1:])
                variables[var] = getattr(obj, 'name', None)
            plot_data = plot_data[list(variables)]
        return (plot_data, variables)

    def map_hue(self, palette=None, order=None, norm=None, saturation=1):
        mapping = HueMapping(self, palette, order, norm, saturation)
        self._hue_map = mapping

    def map_size(self, sizes=None, order=None, norm=None):
        mapping = SizeMapping(self, sizes, order, norm)
        self._size_map = mapping

    def map_style(self, markers=None, dashes=None, order=None):
        mapping = StyleMapping(self, markers, dashes, order)
        self._style_map = mapping

    def iter_data(self, grouping_vars=None, *, reverse=False, from_comp_data=False, by_facet=True, allow_empty=False, dropna=True):
        """Generator for getting subsets of data defined by semantic variables.

        Also injects "col" and "row" into grouping semantics.

        Parameters
        ----------
        grouping_vars : string or list of strings
            Semantic variables that define the subsets of data.
        reverse : bool
            If True, reverse the order of iteration.
        from_comp_data : bool
            If True, use self.comp_data rather than self.plot_data
        by_facet : bool
            If True, add faceting variables to the set of grouping variables.
        allow_empty : bool
            If True, yield an empty dataframe when no observations exist for
            combinations of grouping variables.
        dropna : bool
            If True, remove rows with missing data.

        Yields
        ------
        sub_vars : dict
            Keys are semantic names, values are the level of that semantic.
        sub_data : :class:`pandas.DataFrame`
            Subset of ``plot_data`` for this combination of semantic values.

        """
        if grouping_vars is None:
            grouping_vars = []
        elif isinstance(grouping_vars, str):
            grouping_vars = [grouping_vars]
        elif isinstance(grouping_vars, tuple):
            grouping_vars = list(grouping_vars)
        if by_facet:
            facet_vars = {'col', 'row'}
            grouping_vars.extend(facet_vars & set(self.variables) - set(grouping_vars))
        grouping_vars = [var for var in grouping_vars if var in self.variables]
        if from_comp_data:
            data = self.comp_data
        else:
            data = self.plot_data
        if dropna:
            data = data.dropna()
        levels = self.var_levels.copy()
        if from_comp_data:
            for axis in {'x', 'y'} & set(grouping_vars):
                converter = self.converters[axis].iloc[0]
                if self.var_types[axis] == 'categorical':
                    if self._var_ordered[axis]:
                        levels[axis] = converter.convert_units(levels[axis])
                    else:
                        levels[axis] = np.sort(data[axis].unique())
                else:
                    transform = converter.get_transform().transform
                    levels[axis] = transform(converter.convert_units(levels[axis]))
        if grouping_vars:
            grouped_data = data.groupby(grouping_vars, sort=False, as_index=False, observed=False)
            grouping_keys = []
            for var in grouping_vars:
                key = levels.get(var)
                grouping_keys.append([] if key is None else key)
            iter_keys = itertools.product(*grouping_keys)
            if reverse:
                iter_keys = reversed(list(iter_keys))
            for key in iter_keys:
                pd_key = key[0] if len(key) == 1 and _version_predates(pd, '2.2.0') else key
                try:
                    data_subset = grouped_data.get_group(pd_key)
                except KeyError:
                    data_subset = data.loc[[]]
                if data_subset.empty and (not allow_empty):
                    continue
                sub_vars = dict(zip(grouping_vars, key))
                yield (sub_vars, data_subset.copy())
        else:
            yield ({}, data.copy())

    @property
    def comp_data(self):
        """Dataframe with numeric x and y, after unit conversion and log scaling."""
        if not hasattr(self, 'ax'):
            return self.plot_data
        if not hasattr(self, '_comp_data'):
            comp_data = self.plot_data.copy(deep=False).drop(['x', 'y'], axis=1, errors='ignore')
            for var in 'yx':
                if var not in self.variables:
                    continue
                parts = []
                grouped = self.plot_data[var].groupby(self.converters[var], sort=False)
                for converter, orig in grouped:
                    orig = orig.mask(orig.isin([np.inf, -np.inf]), np.nan)
                    orig = orig.dropna()
                    if var in self.var_levels:
                        orig = orig[orig.isin(self.var_levels[var])]
                    comp = pd.to_numeric(converter.convert_units(orig)).astype(float)
                    transform = converter.get_transform().transform
                    parts.append(pd.Series(transform(comp), orig.index, name=orig.name))
                if parts:
                    comp_col = pd.concat(parts)
                else:
                    comp_col = pd.Series(dtype=float, name=var)
                comp_data.insert(0, var, comp_col)
            self._comp_data = comp_data
        return self._comp_data

    def _get_axes(self, sub_vars):
        """Return an Axes object based on existence of row/col variables."""
        row = sub_vars.get('row', None)
        col = sub_vars.get('col', None)
        if row is not None and col is not None:
            return self.facets.axes_dict[row, col]
        elif row is not None:
            return self.facets.axes_dict[row]
        elif col is not None:
            return self.facets.axes_dict[col]
        elif self.ax is None:
            return self.facets.ax
        else:
            return self.ax

    def _attach(self, obj, allowed_types=None, log_scale=None):
        """Associate the plotter with an Axes manager and initialize its units.

        Parameters
        ----------
        obj : :class:`matplotlib.axes.Axes` or :class:'FacetGrid`
            Structural object that we will eventually plot onto.
        allowed_types : str or list of str
            If provided, raise when either the x or y variable does not have
            one of the declared seaborn types.
        log_scale : bool, number, or pair of bools or numbers
            If not False, set the axes to use log scaling, with the given
            base or defaulting to 10. If a tuple, interpreted as separate
            arguments for the x and y axes.

        """
        from .axisgrid import FacetGrid
        if isinstance(obj, FacetGrid):
            self.ax = None
            self.facets = obj
            ax_list = obj.axes.flatten()
            if obj.col_names is not None:
                self.var_levels['col'] = obj.col_names
            if obj.row_names is not None:
                self.var_levels['row'] = obj.row_names
        else:
            self.ax = obj
            self.facets = None
            ax_list = [obj]
        axis_variables = set('xy').intersection(self.variables)
        if allowed_types is None:
            allowed_types = ['numeric', 'datetime', 'categorical']
        elif isinstance(allowed_types, str):
            allowed_types = [allowed_types]
        for var in axis_variables:
            var_type = self.var_types[var]
            if var_type not in allowed_types:
                err = f'The {var} variable is {var_type}, but one of {allowed_types} is required'
                raise TypeError(err)
        facet_dim = {'x': 'col', 'y': 'row'}
        self.converters = {}
        for var in axis_variables:
            other_var = {'x': 'y', 'y': 'x'}[var]
            converter = pd.Series(index=self.plot_data.index, name=var, dtype=object)
            share_state = getattr(self.facets, f'_share{var}', True)
            if share_state is True or share_state == facet_dim[other_var]:
                converter.loc[:] = getattr(ax_list[0], f'{var}axis')
            elif share_state is False:
                for axes_vars, axes_data in self.iter_data():
                    ax = self._get_axes(axes_vars)
                    converter.loc[axes_data.index] = getattr(ax, f'{var}axis')
            else:
                names = getattr(self.facets, f'{share_state}_names')
                for i, level in enumerate(names):
                    idx = (i, 0) if share_state == 'row' else (0, i)
                    axis = getattr(self.facets.axes[idx], f'{var}axis')
                    converter.loc[self.plot_data[share_state] == level] = axis
            self.converters[var] = converter
            grouped = self.plot_data[var].groupby(self.converters[var], sort=False)
            for converter, seed_data in grouped:
                if self.var_types[var] == 'categorical':
                    if self._var_ordered[var]:
                        order = self.var_levels[var]
                    else:
                        order = None
                    seed_data = categorical_order(seed_data, order)
                converter.update_units(seed_data)
        if log_scale is None:
            scalex = scaley = False
        else:
            try:
                scalex, scaley = log_scale
            except TypeError:
                scalex = log_scale if self.var_types.get('x') == 'numeric' else False
                scaley = log_scale if self.var_types.get('y') == 'numeric' else False
        for axis, scale in zip('xy', (scalex, scaley)):
            if scale:
                for ax in ax_list:
                    set_scale = getattr(ax, f'set_{axis}scale')
                    if scale is True:
                        set_scale('log', nonpositive='mask')
                    else:
                        set_scale('log', base=scale, nonpositive='mask')
        if self.var_types.get('y', None) == 'categorical':
            for ax in ax_list:
                ax.yaxis.set_inverted(True)

    def _get_scale_transforms(self, axis):
        """Return a function implementing the scale transform (or its inverse)."""
        if self.ax is None:
            axis_list = [getattr(ax, f'{axis}axis') for ax in self.facets.axes.flat]
            scales = {axis.get_scale() for axis in axis_list}
            if len(scales) > 1:
                err = 'Cannot determine transform with mixed scales on faceted axes.'
                raise RuntimeError(err)
            transform_obj = axis_list[0].get_transform()
        else:
            transform_obj = getattr(self.ax, f'{axis}axis').get_transform()
        return (transform_obj.transform, transform_obj.inverted().transform)

    def _add_axis_labels(self, ax, default_x='', default_y=''):
        """Add axis labels if not present, set visibility to match ticklabels."""
        if not ax.get_xlabel():
            x_visible = any((t.get_visible() for t in ax.get_xticklabels()))
            ax.set_xlabel(self.variables.get('x', default_x), visible=x_visible)
        if not ax.get_ylabel():
            y_visible = any((t.get_visible() for t in ax.get_yticklabels()))
            ax.set_ylabel(self.variables.get('y', default_y), visible=y_visible)

    def add_legend_data(self, ax, func, common_kws=None, attrs=None, semantic_kws=None):
        """Add labeled artists to represent the different plot semantics."""
        verbosity = self.legend
        if isinstance(verbosity, str) and verbosity not in ['auto', 'brief', 'full']:
            err = "`legend` must be 'auto', 'brief', 'full', or a boolean."
            raise ValueError(err)
        elif verbosity is True:
            verbosity = 'auto'
        keys = []
        legend_kws = {}
        common_kws = {} if common_kws is None else common_kws.copy()
        semantic_kws = {} if semantic_kws is None else semantic_kws.copy()
        titles = {title for title in (self.variables.get(v, None) for v in ['hue', 'size', 'style']) if title is not None}
        title = '' if len(titles) != 1 else titles.pop()
        title_kws = dict(visible=False, color='w', s=0, linewidth=0, marker='', dashes='')

        def update(var_name, val_name, **kws):
            key = (var_name, val_name)
            if key in legend_kws:
                legend_kws[key].update(**kws)
            else:
                keys.append(key)
                legend_kws[key] = dict(**kws)
        if attrs is None:
            attrs = {'hue': 'color', 'size': ['linewidth', 's'], 'style': None}
        for var, names in attrs.items():
            self._update_legend_data(update, var, verbosity, title, title_kws, names, semantic_kws.get(var))
        legend_data = {}
        legend_order = []
        if common_kws.get('color', False) is None:
            common_kws.pop('color')
        for key in keys:
            _, label = key
            kws = legend_kws[key]
            level_kws = {}
            use_attrs = [*self._legend_attributes, *common_kws, *[attr for var_attrs in semantic_kws.values() for attr in var_attrs]]
            for attr in use_attrs:
                if attr in kws:
                    level_kws[attr] = kws[attr]
            artist = func(label=label, **{'color': '.2', **common_kws, **level_kws})
            if _version_predates(mpl, '3.5.0'):
                if isinstance(artist, mpl.lines.Line2D):
                    ax.add_line(artist)
                elif isinstance(artist, mpl.patches.Patch):
                    ax.add_patch(artist)
                elif isinstance(artist, mpl.collections.Collection):
                    ax.add_collection(artist)
            else:
                ax.add_artist(artist)
            legend_data[key] = artist
            legend_order.append(key)
        self.legend_title = title
        self.legend_data = legend_data
        self.legend_order = legend_order

    def _update_legend_data(self, update, var, verbosity, title, title_kws, attr_names, other_props):
        """Generate legend tick values and formatted labels."""
        brief_ticks = 6
        mapper = getattr(self, f'_{var}_map', None)
        if mapper is None:
            return
        brief = mapper.map_type == 'numeric' and (verbosity == 'brief' or (verbosity == 'auto' and len(mapper.levels) > brief_ticks))
        if brief:
            if isinstance(mapper.norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=brief_ticks)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=brief_ticks)
            limits = (min(mapper.levels), max(mapper.levels))
            levels, formatted_levels = locator_to_legend_entries(locator, limits, self.plot_data[var].infer_objects().dtype)
        elif mapper.levels is None:
            levels = formatted_levels = []
        else:
            levels = formatted_levels = mapper.levels
        if not title and self.variables.get(var, None) is not None:
            update((self.variables[var], 'title'), self.variables[var], **title_kws)
        other_props = {} if other_props is None else other_props
        for level, formatted_level in zip(levels, formatted_levels):
            if level is not None:
                attr = mapper(level)
                if isinstance(attr_names, list):
                    attr = {name: attr for name in attr_names}
                elif attr_names is not None:
                    attr = {attr_names: attr}
                attr.update({k: v[level] for k, v in other_props.items() if level in v})
                update(self.variables[var], formatted_level, **attr)

    def scale_native(self, axis, *args, **kwargs):
        raise NotImplementedError

    def scale_numeric(self, axis, *args, **kwargs):
        raise NotImplementedError

    def scale_datetime(self, axis, *args, **kwargs):
        raise NotImplementedError

    def scale_categorical(self, axis, order=None, formatter=None):
        """
        Enforce categorical (fixed-scale) rules for the data on given axis.

        Parameters
        ----------
        axis : "x" or "y"
            Axis of the plot to operate on.
        order : list
            Order that unique values should appear in.
        formatter : callable
            Function mapping values to a string representation.

        Returns
        -------
        self

        """
        _check_argument('axis', ['x', 'y'], axis)
        if axis not in self.variables:
            self.variables[axis] = None
            self.var_types[axis] = 'categorical'
            self.plot_data[axis] = ''
        if self.var_types[axis] == 'numeric':
            self.plot_data = self.plot_data.sort_values(axis, kind='mergesort')
        cat_data = self.plot_data[axis].dropna()
        self._var_ordered[axis] = order is not None or cat_data.dtype.name == 'category'
        order = pd.Index(categorical_order(cat_data, order), name=axis)
        if formatter is not None:
            cat_data = cat_data.map(formatter)
            order = order.map(formatter)
        else:
            cat_data = cat_data.astype(str)
            order = order.astype(str)
        self.var_levels[axis] = order
        self.var_types[axis] = 'categorical'
        self.plot_data[axis] = cat_data
        return self