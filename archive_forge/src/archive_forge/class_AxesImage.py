import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
class AxesImage(_ImageBase):
    """
    An image attached to an Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The axes the image will belong to.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar
        data to colors.
    norm : str or `~matplotlib.colors.Normalize`
        Maps luminance to 0-1.
    interpolation : str, default: :rc:`image.interpolation`
        Supported values are 'none', 'antialiased', 'nearest', 'bilinear',
        'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
        'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
        'sinc', 'lanczos', 'blackman'.
    interpolation_stage : {'data', 'rgba'}, default: 'data'
        If 'data', interpolation
        is carried out on the data provided by the user.  If 'rgba', the
        interpolation is carried out after the colormapping has been
        applied (visual interpolation).
    origin : {'upper', 'lower'}, default: :rc:`image.origin`
        Place the [0, 0] index of the array in the upper left or lower left
        corner of the axes. The convention 'upper' is typically used for
        matrices and images.
    extent : tuple, optional
        The data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.
    filternorm : bool, default: True
        A parameter for the antigrain image resize filter
        (see the antigrain documentation).
        If filternorm is set, the filter normalizes integer values and corrects
        the rounding errors. It doesn't do anything with the source floating
        point values, it corrects only integers according to the rule of 1.0
        which means that any sum of pixel weights must be equal to 1.0. So,
        the filter function must produce a graph of the proper shape.
    filterrad : float > 0, default: 4
        The filter radius for filters that have a radius parameter, i.e. when
        interpolation is one of: 'sinc', 'lanczos' or 'blackman'.
    resample : bool, default: False
        When True, use a full resampling method. When False, only resample when
        the output image is larger than the input image.
    **kwargs : `~matplotlib.artist.Artist` properties
    """

    def __init__(self, ax, *, cmap=None, norm=None, interpolation=None, origin=None, extent=None, filternorm=True, filterrad=4.0, resample=False, interpolation_stage=None, **kwargs):
        self._extent = extent
        super().__init__(ax, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, filternorm=filternorm, filterrad=filterrad, resample=resample, interpolation_stage=interpolation_stage, **kwargs)

    def get_window_extent(self, renderer=None):
        x0, x1, y0, y1 = self._extent
        bbox = Bbox.from_extents([x0, y0, x1, y1])
        return bbox.transformed(self.get_transform())

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        trans = self.get_transform()
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        clip = self.get_clip_box() or self.axes.bbox if self.get_clip_on() else self.figure.bbox
        return self._make_image(self._A, bbox, transformed_bbox, clip, magnification, unsampled=unsampled)

    def _check_unsampled_image(self):
        """Return whether the image would be better drawn unsampled."""
        return self.get_interpolation() == 'none'

    def set_extent(self, extent, **kwargs):
        """
        Set the image extent.

        Parameters
        ----------
        extent : 4-tuple of float
            The position and size of the image as tuple
            ``(left, right, bottom, top)`` in data coordinates.
        **kwargs
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D axes), *runits* and *thetaunits* (for
            polar axes) entries are applied, if present.

        Notes
        -----
        This updates ``ax.dataLim``, and, if autoscaling, sets ``ax.viewLim``
        to tightly fit the image, regardless of ``dataLim``.  Autoscaling
        state is not changed, so following this with ``ax.autoscale_view()``
        will redo the autoscaling in accord with ``dataLim``.
        """
        (xmin, xmax), (ymin, ymax) = self.axes._process_unit_info([('x', [extent[0], extent[1]]), ('y', [extent[2], extent[3]])], kwargs)
        if kwargs:
            raise _api.kwarg_error('set_extent', kwargs)
        xmin = self.axes._validate_converted_limits(xmin, self.convert_xunits)
        xmax = self.axes._validate_converted_limits(xmax, self.convert_xunits)
        ymin = self.axes._validate_converted_limits(ymin, self.convert_yunits)
        ymax = self.axes._validate_converted_limits(ymax, self.convert_yunits)
        extent = [xmin, xmax, ymin, ymax]
        self._extent = extent
        corners = ((xmin, ymin), (xmax, ymax))
        self.axes.update_datalim(corners)
        self.sticky_edges.x[:] = [xmin, xmax]
        self.sticky_edges.y[:] = [ymin, ymax]
        if self.axes.get_autoscalex_on():
            self.axes.set_xlim((xmin, xmax), auto=None)
        if self.axes.get_autoscaley_on():
            self.axes.set_ylim((ymin, ymax), auto=None)
        self.stale = True

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
        if self._extent is not None:
            return self._extent
        else:
            sz = self.get_size()
            numrows, numcols = sz
            if self.origin == 'upper':
                return (-0.5, numcols - 0.5, numrows - 0.5, -0.5)
            else:
                return (-0.5, numcols - 0.5, -0.5, numrows - 0.5)

    def get_cursor_data(self, event):
        """
        Return the image value at the event position or *None* if the event is
        outside the image.

        See Also
        --------
        matplotlib.artist.Artist.get_cursor_data
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == 'upper':
            ymin, ymax = (ymax, ymin)
        arr = self.get_array()
        data_extent = Bbox([[xmin, ymin], [xmax, ymax]])
        array_extent = Bbox([[0, 0], [arr.shape[1], arr.shape[0]]])
        trans = self.get_transform().inverted()
        trans += BboxTransform(boxin=data_extent, boxout=array_extent)
        point = trans.transform([event.x, event.y])
        if any(np.isnan(point)):
            return None
        j, i = point.astype(int)
        if not 0 <= i < arr.shape[0] or not 0 <= j < arr.shape[1]:
            return None
        else:
            return arr[i, j]