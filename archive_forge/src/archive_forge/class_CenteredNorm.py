import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
class CenteredNorm(Normalize):

    def __init__(self, vcenter=0, halfrange=None, clip=False):
        """
        Normalize symmetrical data around a center (0 by default).

        Unlike `TwoSlopeNorm`, `CenteredNorm` applies an equal rate of change
        around the center.

        Useful when mapping symmetrical data around a conceptual center
        e.g., data that range from -2 to 4, with 0 as the midpoint, and
        with equal rates of change around that midpoint.

        Parameters
        ----------
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are also
            transformed, resulting in values outside ``[0, 1]``. For a
            standard use with colormaps, this behavior is desired because colormaps
            mark these outside values with specific colors for *over* or *under*.

            If ``True`` values falling outside the range ``[vmin, vmax]``,
            are mapped to 0 or 1, whichever is closer. This makes these values
            indistinguishable from regular boundary values and can lead to
            misinterpretation of the data.

        Examples
        --------
        This maps data values -2 to 0.25, 0 to 0.5, and 4 to 1.0
        (assuming equal rates of change above and below 0.0):

            >>> import matplotlib.colors as mcolors
            >>> norm = mcolors.CenteredNorm(halfrange=4.0)
            >>> data = [-2., 0., 4.]
            >>> norm(data)
            array([0.25, 0.5 , 1.  ])
        """
        super().__init__(vmin=None, vmax=None, clip=clip)
        self._vcenter = vcenter
        self.halfrange = halfrange

    def autoscale(self, A):
        """
        Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
        """
        A = np.asanyarray(A)
        self.halfrange = max(self._vcenter - A.min(), A.max() - self._vcenter)

    def autoscale_None(self, A):
        """Set *vmin* and *vmax*."""
        A = np.asanyarray(A)
        if self.halfrange is None and A.size:
            self.autoscale(A)

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmin:
            self._vmin = value
            self._vmax = 2 * self.vcenter - value
            self._changed()

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        value = _sanitize_extrema(value)
        if value != self._vmax:
            self._vmax = value
            self._vmin = 2 * self.vcenter - value
            self._changed()

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, vcenter):
        if vcenter != self._vcenter:
            self._vcenter = vcenter
            self.halfrange = self.halfrange
            self._changed()

    @property
    def halfrange(self):
        if self.vmin is None or self.vmax is None:
            return None
        return (self.vmax - self.vmin) / 2

    @halfrange.setter
    def halfrange(self, halfrange):
        if halfrange is None:
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = self.vcenter - abs(halfrange)
            self.vmax = self.vcenter + abs(halfrange)