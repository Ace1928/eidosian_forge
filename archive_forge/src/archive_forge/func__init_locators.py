from numbers import Number
import functools
from types import MethodType
import numpy as np
from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec
from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist
def _init_locators(self):
    if self._colorbar_pad is None:
        if self._colorbar_location in ('left', 'right'):
            self._colorbar_pad = self._horiz_pad_size.fixed_size
        else:
            self._colorbar_pad = self._vert_pad_size.fixed_size
    self.cbar_axes = [_cbaraxes_class_factory(self._defaultAxesClass)(self.axes_all[0].figure, self._divider.get_position(), orientation=self._colorbar_location) for _ in range(self.ngrids)]
    cb_mode = self._colorbar_mode
    cb_location = self._colorbar_location
    h = []
    v = []
    h_ax_pos = []
    h_cb_pos = []
    if cb_mode == 'single' and cb_location in ('left', 'bottom'):
        if cb_location == 'left':
            sz = self._nrows * Size.AxesX(self.axes_llc)
            h.append(Size.from_any(self._colorbar_size, sz))
            h.append(Size.from_any(self._colorbar_pad, sz))
            locator = self._divider.new_locator(nx=0, ny=0, ny1=-1)
        elif cb_location == 'bottom':
            sz = self._ncols * Size.AxesY(self.axes_llc)
            v.append(Size.from_any(self._colorbar_size, sz))
            v.append(Size.from_any(self._colorbar_pad, sz))
            locator = self._divider.new_locator(nx=0, nx1=-1, ny=0)
        for i in range(self.ngrids):
            self.cbar_axes[i].set_visible(False)
        self.cbar_axes[0].set_axes_locator(locator)
        self.cbar_axes[0].set_visible(True)
    for col, ax in enumerate(self.axes_row[0]):
        if h:
            h.append(self._horiz_pad_size)
        if ax:
            sz = Size.AxesX(ax, aspect='axes', ref_ax=self.axes_all[0])
        else:
            sz = Size.AxesX(self.axes_all[0], aspect='axes', ref_ax=self.axes_all[0])
        if cb_location == 'left' and (cb_mode == 'each' or (cb_mode == 'edge' and col == 0)):
            h_cb_pos.append(len(h))
            h.append(Size.from_any(self._colorbar_size, sz))
            h.append(Size.from_any(self._colorbar_pad, sz))
        h_ax_pos.append(len(h))
        h.append(sz)
        if cb_location == 'right' and (cb_mode == 'each' or (cb_mode == 'edge' and col == self._ncols - 1)):
            h.append(Size.from_any(self._colorbar_pad, sz))
            h_cb_pos.append(len(h))
            h.append(Size.from_any(self._colorbar_size, sz))
    v_ax_pos = []
    v_cb_pos = []
    for row, ax in enumerate(self.axes_column[0][::-1]):
        if v:
            v.append(self._vert_pad_size)
        if ax:
            sz = Size.AxesY(ax, aspect='axes', ref_ax=self.axes_all[0])
        else:
            sz = Size.AxesY(self.axes_all[0], aspect='axes', ref_ax=self.axes_all[0])
        if cb_location == 'bottom' and (cb_mode == 'each' or (cb_mode == 'edge' and row == 0)):
            v_cb_pos.append(len(v))
            v.append(Size.from_any(self._colorbar_size, sz))
            v.append(Size.from_any(self._colorbar_pad, sz))
        v_ax_pos.append(len(v))
        v.append(sz)
        if cb_location == 'top' and (cb_mode == 'each' or (cb_mode == 'edge' and row == self._nrows - 1)):
            v.append(Size.from_any(self._colorbar_pad, sz))
            v_cb_pos.append(len(v))
            v.append(Size.from_any(self._colorbar_size, sz))
    for i in range(self.ngrids):
        col, row = self._get_col_row(i)
        locator = self._divider.new_locator(nx=h_ax_pos[col], ny=v_ax_pos[self._nrows - 1 - row])
        self.axes_all[i].set_axes_locator(locator)
        if cb_mode == 'each':
            if cb_location in ('right', 'left'):
                locator = self._divider.new_locator(nx=h_cb_pos[col], ny=v_ax_pos[self._nrows - 1 - row])
            elif cb_location in ('top', 'bottom'):
                locator = self._divider.new_locator(nx=h_ax_pos[col], ny=v_cb_pos[self._nrows - 1 - row])
            self.cbar_axes[i].set_axes_locator(locator)
        elif cb_mode == 'edge':
            if cb_location == 'left' and col == 0 or (cb_location == 'right' and col == self._ncols - 1):
                locator = self._divider.new_locator(nx=h_cb_pos[0], ny=v_ax_pos[self._nrows - 1 - row])
                self.cbar_axes[row].set_axes_locator(locator)
            elif cb_location == 'bottom' and row == self._nrows - 1 or (cb_location == 'top' and row == 0):
                locator = self._divider.new_locator(nx=h_ax_pos[col], ny=v_cb_pos[0])
                self.cbar_axes[col].set_axes_locator(locator)
    if cb_mode == 'single':
        if cb_location == 'right':
            sz = self._nrows * Size.AxesX(self.axes_llc)
            h.append(Size.from_any(self._colorbar_pad, sz))
            h.append(Size.from_any(self._colorbar_size, sz))
            locator = self._divider.new_locator(nx=-2, ny=0, ny1=-1)
        elif cb_location == 'top':
            sz = self._ncols * Size.AxesY(self.axes_llc)
            v.append(Size.from_any(self._colorbar_pad, sz))
            v.append(Size.from_any(self._colorbar_size, sz))
            locator = self._divider.new_locator(nx=0, nx1=-1, ny=-2)
        if cb_location in ('right', 'top'):
            for i in range(self.ngrids):
                self.cbar_axes[i].set_visible(False)
            self.cbar_axes[0].set_axes_locator(locator)
            self.cbar_axes[0].set_visible(True)
    elif cb_mode == 'each':
        for i in range(self.ngrids):
            self.cbar_axes[i].set_visible(True)
    elif cb_mode == 'edge':
        if cb_location in ('right', 'left'):
            count = self._nrows
        else:
            count = self._ncols
        for i in range(count):
            self.cbar_axes[i].set_visible(True)
        for j in range(i + 1, self.ngrids):
            self.cbar_axes[j].set_visible(False)
    else:
        for i in range(self.ngrids):
            self.cbar_axes[i].set_visible(False)
            self.cbar_axes[i].set_position([1.0, 1.0, 0.001, 0.001], which='active')
    self._divider.set_horizontal(h)
    self._divider.set_vertical(v)