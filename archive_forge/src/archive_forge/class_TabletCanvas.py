import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class TabletCanvas(EventDispatcher):
    """Event dispatcher for tablets.

    Use `Tablet.open` to obtain this object for a particular tablet device and
    window.  Events may be generated even if the tablet stylus is outside of
    the window; this is operating-system dependent.

    The events each provide the `TabletCursor` that was used to generate the
    event; for example, to distinguish between a stylus and an eraser.  Only
    one cursor can be used at a time, otherwise the results are undefined.

    :Ivariables:
        `window` : Window
            The window on which this tablet was opened.
    """

    def __init__(self, window):
        self.window = window

    def close(self):
        """Close the tablet device for this window.
        """
        raise NotImplementedError('abstract')
    if _is_pyglet_doc_run:

        def on_enter(self, cursor):
            """A cursor entered the proximity of the window.  The cursor may
            be hovering above the tablet surface, but outside of the window
            bounds, or it may have entered the window bounds.

            Note that you cannot rely on `on_enter` and `on_leave` events to
            be generated in pairs; some events may be lost if the cursor was
            out of the window bounds at the time.

            :Parameters:
                `cursor` : `TabletCursor`
                    The cursor that entered proximity.

            :event:
            """

        def on_leave(self, cursor):
            """A cursor left the proximity of the window.  The cursor may have
            moved too high above the tablet surface to be detected, or it may
            have left the bounds of the window.

            Note that you cannot rely on `on_enter` and `on_leave` events to
            be generated in pairs; some events may be lost if the cursor was
            out of the window bounds at the time.

            :Parameters:
                `cursor` : `TabletCursor`
                    The cursor that left proximity.

            :event:
            """

        def on_motion(self, cursor, x, y, pressure, tilt_x, tilt_y, buttons):
            """The cursor moved on the tablet surface.

            If `pressure` is 0, then the cursor is actually hovering above the
            tablet surface, not in contact.

            :Parameters:
                `cursor` : `TabletCursor`
                    The cursor that moved.
                `x` : int
                    The X position of the cursor, in window coordinates.
                `y` : int
                    The Y position of the cursor, in window coordinates.
                `pressure` : float
                    The pressure applied to the cursor, in range 0.0 (no
                    pressure) to 1.0 (full pressure).
                `tilt_x` : float
                    Currently undefined.
                `tilt_y` : float
                    Currently undefined.
                `buttons` : int
                    Button state may be provided if the platform supports it.
                    Supported on: Windows

            :event:
            """