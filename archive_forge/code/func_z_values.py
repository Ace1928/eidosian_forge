from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
def z_values(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, ax: Axes | int=0, color: str='green', fmt: str='.1f', quad_as_tri: bool=False) -> None:
    """Show ``z`` values on a single Axes.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (array-like of shape (ny, nx): z-values.
            ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
            color (str, optional): Color of added text. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``tab10`` colormap. Default ``"green"``.
            fmt (str, optional): Format to display z-values, default ``".1f"``.
            quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centers
                of quads.

        Warning:
            ``quad_as_tri=True`` shows z-values for all quads, even if masked.
        """
    ax = self._get_ax(ax)
    x, y = self._grid_as_2d(x, y)
    z = np.asarray(z)
    ny, nx = z.shape
    for j in range(ny):
        for i in range(nx):
            ax.text(x[j, i], y[j, i], f'{z[j, i]:{fmt}}', ha='center', va='center', color=color, clip_on=True)
    if quad_as_tri:
        for j in range(ny - 1):
            for i in range(nx - 1):
                xx = np.mean(x[j:j + 2, i:i + 2])
                yy = np.mean(y[j:j + 2, i:i + 2])
                zz = np.mean(z[j:j + 2, i:i + 2])
                ax.text(xx, yy, f'{zz:{fmt}}', ha='center', va='center', color=color, clip_on=True)