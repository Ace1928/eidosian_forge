from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _draw_rasterized(figure, artists, renderer):
    """
    A helper function for rasterizing the list of artists.

    The bookkeeping to track if we are or are not in rasterizing mode
    with the mixed-mode backends is relatively complicated and is now
    handled in the matplotlib.artist.allow_rasterization decorator.

    This helper defines the absolute minimum methods and attributes on a
    shim class to be compatible with that decorator and then uses it to
    rasterize the list of artists.

    This is maybe too-clever, but allows us to re-use the same code that is
    used on normal artists to participate in the "are we rasterizing"
    accounting.

    Please do not use this outside of the "rasterize below a given zorder"
    functionality of Axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure all of the artists belong to (not checked).  We need this
        because we can at the figure level suppress composition and insert each
        rasterized artist as its own image.

    artists : List[matplotlib.artist.Artist]
        The list of Artists to be rasterized.  These are assumed to all
        be in the same Figure.

    renderer : matplotlib.backendbases.RendererBase
        The currently active renderer

    Returns
    -------
    None

    """

    class _MinimalArtist:

        def get_rasterized(self):
            return True

        def get_agg_filter(self):
            return None

        def __init__(self, figure, artists):
            self.figure = figure
            self.artists = artists

        @martist.allow_rasterization
        def draw(self, renderer):
            for a in self.artists:
                a.draw(renderer)
    return _MinimalArtist(figure, artists).draw(renderer)