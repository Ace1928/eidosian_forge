from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _get_draw_artists(self, renderer):
    """Also runs apply_aspect"""
    artists = self.get_children()
    for sfig in self.subfigs:
        artists.remove(sfig)
        childa = sfig.get_children()
        for child in childa:
            if child in artists:
                artists.remove(child)
    artists.remove(self.patch)
    artists = sorted((artist for artist in artists if not artist.get_animated()), key=lambda artist: artist.get_zorder())
    for ax in self._localaxes:
        locator = ax.get_axes_locator()
        ax.apply_aspect(locator(ax, renderer) if locator else None)
        for child in ax.get_children():
            if hasattr(child, 'apply_aspect'):
                locator = child.get_axes_locator()
                child.apply_aspect(locator(child, renderer) if locator else None)
    return artists