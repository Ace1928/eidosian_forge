import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def add_wms(self, wms, layers, wms_kwargs=None, **kwargs):
    """
        Add the specified WMS layer to the axes.

        This function requires owslib and PIL to work.

        Parameters
        ----------
        wms: string or :class:`owslib.wms.WebMapService` instance
            The web map service URL or owslib WMS instance to use.
        layers: string or iterable of string
            The name of the layer(s) to use.
        wms_kwargs: dict or None, optional
            Passed through to the
            :class:`~cartopy.io.ogc_clients.WMSRasterSource`
            constructor's ``getmap_extra_kwargs`` for defining
            getmap time keyword arguments.


        All other keywords are passed through to the construction of the
        image artist. See :meth:`~matplotlib.axes.Axes.imshow()` for
        more details.

        """
    from cartopy.io.ogc_clients import WMSRasterSource
    wms = WMSRasterSource(wms, layers, getmap_extra_kwargs=wms_kwargs)
    return self.add_raster(wms, **kwargs)