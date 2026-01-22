from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def allow_rasterization(draw):
    """
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependent renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    """

    @wraps(draw)
    def draw_wrapper(artist, renderer):
        try:
            if artist.get_rasterized():
                if renderer._raster_depth == 0 and (not renderer._rasterizing):
                    renderer.start_rasterizing()
                    renderer._rasterizing = True
                renderer._raster_depth += 1
            elif renderer._raster_depth == 0 and renderer._rasterizing:
                renderer.stop_rasterizing()
                renderer._rasterizing = False
            if artist.get_agg_filter() is not None:
                renderer.start_filter()
            return draw(artist, renderer)
        finally:
            if artist.get_agg_filter() is not None:
                renderer.stop_filter(artist.get_agg_filter())
            if artist.get_rasterized():
                renderer._raster_depth -= 1
            if renderer._rasterizing and artist.figure and artist.figure.suppressComposite:
                renderer.stop_rasterizing()
                renderer.start_rasterizing()
    draw_wrapper._supports_rasterization = True
    return draw_wrapper