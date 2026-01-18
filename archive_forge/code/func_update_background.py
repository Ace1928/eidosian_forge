from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def update_background(self, event):
    """Force an update of the background."""
    if not self.useblit:
        return
    artists = sorted(self.artists + self._get_animated_artists(), key=lambda a: a.get_zorder())
    needs_redraw = any((artist.get_visible() for artist in artists))
    with ExitStack() as stack:
        if needs_redraw:
            for artist in artists:
                stack.enter_context(artist._cm_set(visible=False))
            self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
    if needs_redraw:
        for artist in artists:
            self.ax.draw_artist(artist)