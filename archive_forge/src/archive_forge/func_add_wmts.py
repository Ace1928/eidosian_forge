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
def add_wmts(self, wmts, layer_name, wmts_kwargs=None, **kwargs):
    """
        Add the specified WMTS layer to the axes.

        This function requires owslib and PIL to work.

        Parameters
        ----------
        wmts
            The URL of the WMTS, or an owslib.wmts.WebMapTileService instance.
        layer_name
            The name of the layer to use.
        wmts_kwargs: dict or None, optional
            Passed through to the
            :class:`~cartopy.io.ogc_clients.WMTSRasterSource` constructor's
            ``gettile_extra_kwargs`` (e.g. time).


        All other keywords are passed through to the construction of the
        image artist. See :meth:`~matplotlib.axes.Axes.imshow()` for
        more details.

        """
    from cartopy.io.ogc_clients import WMTSRasterSource
    wmts = WMTSRasterSource(wmts, layer_name, gettile_extra_kwargs=wmts_kwargs)
    return self.add_raster(wmts, **kwargs)