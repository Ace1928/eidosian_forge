import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
class DraggableBase:
    """
    Helper base class for a draggable artist (legend, offsetbox).

    Derived classes must override the following methods::

        def save_offset(self):
            '''
            Called when the object is picked for dragging; should save the
            reference position of the artist.
            '''

        def update_offset(self, dx, dy):
            '''
            Called during the dragging; (*dx*, *dy*) is the pixel offset from
            the point where the mouse drag started.
            '''

    Optionally, you may override the following method::

        def finalize_offset(self):
            '''Called when the mouse is released.'''

    In the current implementation of `.DraggableLegend` and
    `DraggableAnnotation`, `update_offset` places the artists in display
    coordinates, and `finalize_offset` recalculates their position in axes
    coordinate and set a relevant attribute.
    """

    def __init__(self, ref_artist, use_blit=False):
        self.ref_artist = ref_artist
        if not ref_artist.pickable():
            ref_artist.set_picker(True)
        self.got_artist = False
        self._use_blit = use_blit and self.canvas.supports_blit
        callbacks = ref_artist.figure._canvas_callbacks
        self._disconnectors = [functools.partial(callbacks.disconnect, callbacks._connect_picklable(name, func)) for name, func in [('pick_event', self.on_pick), ('button_release_event', self.on_release), ('motion_notify_event', self.on_motion)]]
    canvas = property(lambda self: self.ref_artist.figure.canvas)
    cids = property(lambda self: [disconnect.args[0] for disconnect in self._disconnectors[:2]])

    def on_motion(self, evt):
        if self._check_still_parented() and self.got_artist:
            dx = evt.x - self.mouse_x
            dy = evt.y - self.mouse_y
            self.update_offset(dx, dy)
            if self._use_blit:
                self.canvas.restore_region(self.background)
                self.ref_artist.draw(self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            else:
                self.canvas.draw()

    def on_pick(self, evt):
        if self._check_still_parented() and evt.artist == self.ref_artist:
            self.mouse_x = evt.mouseevent.x
            self.mouse_y = evt.mouseevent.y
            self.got_artist = True
            if self._use_blit:
                self.ref_artist.set_animated(True)
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ref_artist.figure.bbox)
                self.ref_artist.draw(self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            self.save_offset()

    def on_release(self, event):
        if self._check_still_parented() and self.got_artist:
            self.finalize_offset()
            self.got_artist = False
            if self._use_blit:
                self.ref_artist.set_animated(False)

    def _check_still_parented(self):
        if self.ref_artist.figure is None:
            self.disconnect()
            return False
        else:
            return True

    def disconnect(self):
        """Disconnect the callbacks."""
        for disconnector in self._disconnectors:
            disconnector()

    def save_offset(self):
        pass

    def update_offset(self, dx, dy):
        pass

    def finalize_offset(self):
        pass