from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
class PairGrid(Grid):
    """Subplot grid for plotting pairwise relationships in a dataset.

    This object maps each variable in a dataset onto a column and row in a
    grid of multiple axes. Different axes-level plotting functions can be
    used to draw bivariate plots in the upper and lower triangles, and the
    marginal distribution of each variable can be shown on the diagonal.

    Several different common plots can be generated in a single line using
    :func:`pairplot`. Use :class:`PairGrid` when you need more flexibility.

    See the :ref:`tutorial <grid_tutorial>` for more information.

    """

    def __init__(self, data, *, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None, corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False):
        """Initialize the plot figure and PairGrid object.

        Parameters
        ----------
        data : DataFrame
            Tidy (long-form) dataframe where each column is a variable and
            each row is an observation.
        hue : string (variable name)
            Variable in ``data`` to map plot aspects to different colors. This
            variable will be excluded from the default x and y variables.
        vars : list of variable names
            Variables within ``data`` to use, otherwise use every column with
            a numeric datatype.
        {x, y}_vars : lists of variable names
            Variables within ``data`` to use separately for the rows and
            columns of the figure; i.e. to make a non-square plot.
        hue_order : list of strings
            Order for the levels of the hue variable in the palette
        palette : dict or seaborn color palette
            Set of colors for mapping the ``hue`` variable. If a dict, keys
            should be values  in the ``hue`` variable.
        hue_kws : dictionary of param -> list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable (e.g.
            the markers in a scatterplot).
        corner : bool
            If True, don't add axes to the upper (off-diagonal) triangle of the
            grid, making this a "corner" plot.
        height : scalar
            Height (in inches) of each facet.
        aspect : scalar
            Aspect * height gives the width (in inches) of each facet.
        layout_pad : scalar
            Padding between axes; passed to ``fig.tight_layout``.
        despine : boolean
            Remove the top and right spines from the plots.
        dropna : boolean
            Drop missing values from the data before plotting.

        See Also
        --------
        pairplot : Easily drawing common uses of :class:`PairGrid`.
        FacetGrid : Subplot grid for plotting conditional relationships.

        Examples
        --------

        .. include:: ../docstrings/PairGrid.rst

        """
        super().__init__()
        data = handle_data_source(data)
        numeric_cols = self._find_numeric_cols(data)
        if hue in numeric_cols:
            numeric_cols.remove(hue)
        if vars is not None:
            x_vars = list(vars)
            y_vars = list(vars)
        if x_vars is None:
            x_vars = numeric_cols
        if y_vars is None:
            y_vars = numeric_cols
        if np.isscalar(x_vars):
            x_vars = [x_vars]
        if np.isscalar(y_vars):
            y_vars = [y_vars]
        self.x_vars = x_vars = list(x_vars)
        self.y_vars = y_vars = list(y_vars)
        self.square_grid = self.x_vars == self.y_vars
        if not x_vars:
            raise ValueError('No variables found for grid columns.')
        if not y_vars:
            raise ValueError('No variables found for grid rows.')
        figsize = (len(x_vars) * height * aspect, len(y_vars) * height)
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        axes = fig.subplots(len(y_vars), len(x_vars), sharex='col', sharey='row', squeeze=False)
        self._corner = corner
        if corner:
            hide_indices = np.triu_indices_from(axes, 1)
            for i, j in zip(*hide_indices):
                axes[i, j].remove()
                axes[i, j] = None
        self._figure = fig
        self.axes = axes
        self.data = data
        self.diag_sharey = diag_sharey
        self.diag_vars = None
        self.diag_axes = None
        self._dropna = dropna
        self._add_axis_labels()
        self._hue_var = hue
        if hue is None:
            self.hue_names = hue_order = ['_nolegend_']
            self.hue_vals = pd.Series(['_nolegend_'] * len(data), index=data.index)
        else:
            hue_names = hue_order = categorical_order(data[hue], hue_order)
            if dropna:
                hue_names = list(filter(pd.notnull, hue_names))
            self.hue_names = hue_names
            self.hue_vals = data[hue]
        self.hue_kws = hue_kws if hue_kws is not None else {}
        self._orig_palette = palette
        self._hue_order = hue_order
        self.palette = self._get_palette(data, hue, hue_order, palette)
        self._legend_data = {}
        for ax in axes[:-1, :].flat:
            if ax is None:
                continue
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.label.set_visible(False)
        for ax in axes[:, 1:].flat:
            if ax is None:
                continue
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.label.set_visible(False)
        self._tight_layout_rect = [0.01, 0.01, 0.99, 0.99]
        self._tight_layout_pad = layout_pad
        self._despine = despine
        if despine:
            utils.despine(fig=fig)
        self.tight_layout(pad=layout_pad)

    def map(self, func, **kwargs):
        """Plot with the same function in every subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        row_indices, col_indices = np.indices(self.axes.shape)
        indices = zip(row_indices.flat, col_indices.flat)
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_lower(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        indices = zip(*np.tril_indices_from(self.axes, -1))
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        indices = zip(*np.triu_indices_from(self.axes, 1))
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_offdiag(self, func, **kwargs):
        """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        if self.square_grid:
            self.map_lower(func, **kwargs)
            if not self._corner:
                self.map_upper(func, **kwargs)
        else:
            indices = []
            for i, y_var in enumerate(self.y_vars):
                for j, x_var in enumerate(self.x_vars):
                    if x_var != y_var:
                        indices.append((i, j))
            self._map_bivariate(func, indices, **kwargs)
        return self

    def map_diag(self, func, **kwargs):
        """Plot with a univariate function on each diagonal subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take an x array as a positional argument and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        if self.diag_axes is None:
            diag_vars = []
            diag_axes = []
            for i, y_var in enumerate(self.y_vars):
                for j, x_var in enumerate(self.x_vars):
                    if x_var == y_var:
                        diag_vars.append(x_var)
                        ax = self.axes[i, j]
                        diag_ax = ax.twinx()
                        diag_ax.set_axis_off()
                        diag_axes.append(diag_ax)
                        if not plt.rcParams.get('ytick.left', True):
                            for tick in ax.yaxis.majorTicks:
                                tick.tick1line.set_visible(False)
                        if self._corner:
                            ax.yaxis.set_visible(False)
                            if self._despine:
                                utils.despine(ax=ax, left=True)
            if self.diag_sharey and diag_axes:
                for ax in diag_axes[1:]:
                    share_axis(diag_axes[0], ax, 'y')
            self.diag_vars = diag_vars
            self.diag_axes = diag_axes
        if 'hue' not in signature(func).parameters:
            return self._map_diag_iter_hue(func, **kwargs)
        for var, ax in zip(self.diag_vars, self.diag_axes):
            plot_kwargs = kwargs.copy()
            if str(func.__module__).startswith('seaborn'):
                plot_kwargs['ax'] = ax
            else:
                plt.sca(ax)
            vector = self.data[var]
            if self._hue_var is not None:
                hue = self.data[self._hue_var]
            else:
                hue = None
            if self._dropna:
                not_na = vector.notna()
                if hue is not None:
                    not_na &= hue.notna()
                vector = vector[not_na]
                if hue is not None:
                    hue = hue[not_na]
            plot_kwargs.setdefault('hue', hue)
            plot_kwargs.setdefault('hue_order', self._hue_order)
            plot_kwargs.setdefault('palette', self._orig_palette)
            func(x=vector, **plot_kwargs)
            ax.legend_ = None
        self._add_axis_labels()
        return self

    def _map_diag_iter_hue(self, func, **kwargs):
        """Put marginal plot on each diagonal axes, iterating over hue."""
        fixed_color = kwargs.pop('color', None)
        for var, ax in zip(self.diag_vars, self.diag_axes):
            hue_grouped = self.data[var].groupby(self.hue_vals, observed=True)
            plot_kwargs = kwargs.copy()
            if str(func.__module__).startswith('seaborn'):
                plot_kwargs['ax'] = ax
            else:
                plt.sca(ax)
            for k, label_k in enumerate(self._hue_order):
                try:
                    data_k = hue_grouped.get_group(label_k)
                except KeyError:
                    data_k = pd.Series([], dtype=float)
                if fixed_color is None:
                    color = self.palette[k]
                else:
                    color = fixed_color
                if self._dropna:
                    data_k = utils.remove_na(data_k)
                if str(func.__module__).startswith('seaborn'):
                    func(x=data_k, label=label_k, color=color, **plot_kwargs)
                else:
                    func(data_k, label=label_k, color=color, **plot_kwargs)
        self._add_axis_labels()
        return self

    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        from .distributions import histplot, kdeplot
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True
        kws = kwargs.copy()
        for i, j in indices:
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:
                continue
            self._plot_bivariate(x_var, y_var, ax, func, **kws)
        self._add_axis_labels()
        if 'hue' in signature(func).parameters:
            self.hue_names = list(self._legend_data)

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot on the specified axes."""
        if 'hue' not in signature(func).parameters:
            self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
            return
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = ax
        else:
            plt.sca(ax)
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]
        if self._hue_var is not None and self._hue_var not in axes_vars:
            axes_vars.append(self._hue_var)
        data = self.data[axes_vars]
        if self._dropna:
            data = data.dropna()
        x = data[x_var]
        y = data[y_var]
        if self._hue_var is None:
            hue = None
        else:
            hue = data.get(self._hue_var)
        if 'hue' not in kwargs:
            kwargs.update({'hue': hue, 'hue_order': self._hue_order, 'palette': self._orig_palette})
        func(x=x, y=y, **kwargs)
        self._update_legend_data(ax)

    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot while iterating over hue subsets."""
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = ax
        else:
            plt.sca(ax)
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]
        hue_grouped = self.data.groupby(self.hue_vals, observed=True)
        for k, label_k in enumerate(self._hue_order):
            kws = kwargs.copy()
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=axes_vars, dtype=float)
            if self._dropna:
                data_k = data_k[axes_vars].dropna()
            x = data_k[x_var]
            y = data_k[y_var]
            for kw, val_list in self.hue_kws.items():
                kws[kw] = val_list[k]
            kws.setdefault('color', self.palette[k])
            if self._hue_var is not None:
                kws['label'] = label_k
            if str(func.__module__).startswith('seaborn'):
                func(x=x, y=y, **kws)
            else:
                func(x, y, **kws)
        self._update_legend_data(ax)

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        for ax, label in zip(self.axes[-1, :], self.x_vars):
            ax.set_xlabel(label)
        for ax, label in zip(self.axes[:, 0], self.y_vars):
            ax.set_ylabel(label)

    def _find_numeric_cols(self, data):
        """Find which variables in a DataFrame are numeric."""
        numeric_cols = []
        for col in data:
            if variable_type(data[col]) == 'numeric':
                numeric_cols.append(col)
        return numeric_cols