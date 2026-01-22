import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        if not isinstance(nrows, Integral) or nrows <= 0:
            raise ValueError(f'Number of rows must be a positive integer, not {nrows!r}')
        if not isinstance(ncols, Integral) or ncols <= 0:
            raise ValueError(f'Number of columns must be a positive integer, not {ncols!r}')
        self._nrows, self._ncols = (nrows, ncols)
        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)

    def __repr__(self):
        height_arg = f', height_ratios={self._row_height_ratios!r}' if len(set(self._row_height_ratios)) != 1 else ''
        width_arg = f', width_ratios={self._col_width_ratios!r}' if len(set(self._col_width_ratios)) != 1 else ''
        return '{clsname}({nrows}, {ncols}{optionals})'.format(clsname=self.__class__.__name__, nrows=self._nrows, ncols=self._ncols, optionals=height_arg + width_arg)
    nrows = property(lambda self: self._nrows, doc='The number of rows in the grid.')
    ncols = property(lambda self: self._ncols, doc='The number of columns in the grid.')

    def get_geometry(self):
        """
        Return a tuple containing the number of rows and columns in the grid.
        """
        return (self._nrows, self._ncols)

    def get_subplot_params(self, figure=None):
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        Create and return a `.SubplotSpec` instance.

        Parameters
        ----------
        loc : (int, int)
            The position of the subplot in the grid as
            ``(row_index, column_index)``.
        rowspan, colspan : int, default: 1
            The number of rows and columns the subplot should span in the grid.
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1 + rowspan, loc2:loc2 + colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        """
        Set the relative widths of the columns.

        *width_ratios* must be of length *ncols*. Each column gets a relative
        width of ``width_ratios[i] / sum(width_ratios)``.
        """
        if width_ratios is None:
            width_ratios = [1] * self._ncols
        elif len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        """
        Return the width ratios.

        This is *None* if no width ratios have been set explicitly.
        """
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
        if height_ratios is None:
            height_ratios = [1] * self._nrows
        elif len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        """
        Return the height ratios.

        This is *None* if no height ratios have been set explicitly.
        """
        return self._row_height_ratios

    @_api.delete_parameter('3.7', 'raw')
    def get_grid_positions(self, fig, raw=False):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.
        raw : bool, default: False
            If *True*, the subplot parameters of the figure are not taken
            into account. The grid spans the range [0, 1] in both directions
            without margins and there is no space between grid cells. This is
            used for constrained_layout.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
        nrows, ncols = self.get_geometry()
        if raw:
            left = 0.0
            right = 1.0
            bottom = 0.0
            top = 1.0
            wspace = 0.0
            hspace = 0.0
        else:
            subplot_params = self.get_subplot_params(fig)
            left = subplot_params.left
            right = subplot_params.right
            bottom = subplot_params.bottom
            top = subplot_params.top
            wspace = subplot_params.wspace
            hspace = subplot_params.hspace
        tot_width = right - left
        tot_height = top - bottom
        cell_h = tot_height / (nrows + hspace * (nrows - 1))
        sep_h = hspace * cell_h
        norm = cell_h * nrows / sum(self._row_height_ratios)
        cell_heights = [r * norm for r in self._row_height_ratios]
        sep_heights = [0] + [sep_h] * (nrows - 1)
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
        cell_w = tot_width / (ncols + wspace * (ncols - 1))
        sep_w = wspace * cell_w
        norm = cell_w * ncols / sum(self._col_width_ratios)
        cell_widths = [r * norm for r in self._col_width_ratios]
        sep_widths = [0] + [sep_w] * (ncols - 1)
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return (fig_bottoms, fig_tops, fig_lefts, fig_rights)

    @staticmethod
    def _check_gridspec_exists(figure, nrows, ncols):
        """
        Check if the figure already has a gridspec with these dimensions,
        or create a new one
        """
        for ax in figure.get_axes():
            gs = ax.get_gridspec()
            if gs is not None:
                if hasattr(gs, 'get_topmost_subplotspec'):
                    gs = gs.get_topmost_subplotspec().get_gridspec()
                if gs.get_geometry() == (nrows, ncols):
                    return gs
        return GridSpec(nrows, ncols, figure=figure)

    def __getitem__(self, key):
        """Create and return a `.SubplotSpec` instance."""
        nrows, ncols = self.get_geometry()

        def _normalize(key, size, axis):
            orig_key = key
            if isinstance(key, slice):
                start, stop, _ = key.indices(size)
                if stop > start:
                    return (start, stop - 1)
                raise IndexError('GridSpec slice would result in no space allocated for subplot')
            else:
                if key < 0:
                    key = key + size
                if 0 <= key < size:
                    return (key, key)
                elif axis is not None:
                    raise IndexError(f'index {orig_key} is out of bounds for axis {axis} with size {size}')
                else:
                    raise IndexError(f'index {orig_key} is out of bounds for GridSpec with size {size}')
        if isinstance(key, tuple):
            try:
                k1, k2 = key
            except ValueError as err:
                raise ValueError('Unrecognized subplot spec') from err
            num1, num2 = np.ravel_multi_index([_normalize(k1, nrows, 0), _normalize(k2, ncols, 1)], (nrows, ncols))
        else:
            num1, num2 = _normalize(key, nrows * ncols, None)
        return SubplotSpec(self, num1, num2)

    def subplots(self, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None):
        """
        Add all subplots specified by this `GridSpec` to its parent figure.

        See `.Figure.subplots` for detailed documentation.
        """
        figure = self.figure
        if figure is None:
            raise ValueError('GridSpec.subplots() only works for GridSpecs created with a parent figure')
        if not isinstance(sharex, str):
            sharex = 'all' if sharex else 'none'
        if not isinstance(sharey, str):
            sharey = 'all' if sharey else 'none'
        _api.check_in_list(['all', 'row', 'col', 'none', False, True], sharex=sharex, sharey=sharey)
        if subplot_kw is None:
            subplot_kw = {}
        subplot_kw = subplot_kw.copy()
        axarr = np.empty((self._nrows, self._ncols), dtype=object)
        for row in range(self._nrows):
            for col in range(self._ncols):
                shared_with = {'none': None, 'all': axarr[0, 0], 'row': axarr[row, 0], 'col': axarr[0, col]}
                subplot_kw['sharex'] = shared_with[sharex]
                subplot_kw['sharey'] = shared_with[sharey]
                axarr[row, col] = figure.add_subplot(self[row, col], **subplot_kw)
        if sharex in ['col', 'all']:
            for ax in axarr.flat:
                ax._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ['row', 'all']:
            for ax in axarr.flat:
                ax._label_outer_yaxis(skip_non_rectangular_axes=True)
        if squeeze:
            return axarr.item() if axarr.size == 1 else axarr.squeeze()
        else:
            return axarr