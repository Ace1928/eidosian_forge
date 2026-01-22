import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
class DSArtist(_ImageBase):

    def __init__(self, ax, df, glyph, aggregator, agg_hook, shade_hook, plot_width, plot_height, x_range, y_range, width_scale, height_scale, origin='lower', interpolation='none', **kwargs):
        super().__init__(ax, origin=origin, interpolation=interpolation, **kwargs)
        self.axes = ax
        self.df = df
        self.glyph = glyph
        self.aggregator = aggregator
        self.agg_hook = agg_hook
        self.shade_hook = shade_hook
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.width_scale = width_scale
        self.height_scale = height_scale
        if x_range is None:
            x_col = glyph.x_label
            x_range = (df[x_col].min(), df[x_col].max())
        if y_range is None:
            y_col = glyph.y_label
            y_range = (df[y_col].min(), df[y_col].max())
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    def aggregate(self, x_range, y_range):
        """Aggregate data in given range to the window dimensions."""
        dims = self.axes.patch.get_window_extent().bounds
        if self.plot_width is None:
            plot_width = int(int(dims[2] + 0.5) * self.width_scale)
        else:
            plot_width = self.plot_width
        if self.plot_height is None:
            plot_height = int(int(dims[3] + 0.5) * self.height_scale)
        else:
            plot_height = self.plot_height
        canvas = Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)
        binned = bypixel(self.df, canvas, self.glyph, self.aggregator)
        return binned

    def shade(self, binned):
        """Convert an aggregate into an RGBA array."""
        raise NotImplementedError

    def make_image(self, renderer, magnification=1.0, unsampled=True):
        """
        Normalize, rescale, and colormap this image's data for rendering using
        *renderer*, with the given *magnification*.

        If *unsampled* is True, the image will not be scaled, but an
        appropriate affine transformation will be returned instead.

        Returns
        -------
        image : (M, N, 4) uint8 array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : Affine2D
            The affine transformation from image to pixel space.
        """
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        trans = self.get_transform()
        transformed_bbox = TransformedBbox(bbox, trans)
        if self.plot_width is not None or self.plot_height is not None or self.width_scale != 1.0 or (self.height_scale != 1.0):
            unsampled = False
        binned = self.aggregate([x1, x2], [y1, y2])
        if self.agg_hook is not None:
            binned = self.agg_hook(binned)
        self.set_ds_data(binned)
        rgba = self.shade(binned)
        if self.shade_hook is not None:
            img = to_ds_image(binned, rgba)
            img = self.shade_hook(img)
            rgba = uint32_to_uint8(img.data)
        self.set_array(rgba)
        return self._make_image(rgba, bbox, transformed_bbox, self.axes.bbox, magnification=magnification, unsampled=unsampled)

    def set_ds_data(self, binned):
        """
        Set the aggregate data for the bounding box currently displayed.
        Should be a :class:`xarray.DataArray`.
        """
        self._ds_data = binned

    def get_ds_data(self):
        """
        Return the aggregated, pre-shaded :class:`xarray.DataArray` backing the
        bounding box currently displayed.
        """
        return self._ds_data

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)"""
        (x1, x2), (y1, y2) = (self.axes.get_xlim(), self.axes.get_ylim())
        return (x1, x2, y1, y2)

    def get_cursor_data(self, event):
        """
        Return the aggregated data at the event position or *None* if the
        event is outside the bounds of the current view.
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == 'upper':
            ymin, ymax = (ymax, ymin)
        arr = self.get_ds_data().data
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[0, 0], arr.shape[:2]])
        trans = BboxTransform(boxin=data_extent, boxout=array_extent)
        y, x = (event.ydata, event.xdata)
        i, j = trans.transform_point([y, x]).astype(int)
        if not 0 <= i < arr.shape[0] or not 0 <= j < arr.shape[1]:
            return None
        else:
            return arr[i, j]