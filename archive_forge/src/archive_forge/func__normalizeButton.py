from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _normalizeButton(button):
    """
    The left, middle, and right mouse buttons are button numbers 1, 2, and 3 respectively. This is the numbering that
    Xlib on Linux uses (while Windows and macOS don't care about numbers; they just use "left" and "right").

    This function takes one of ``LEFT``, ``MIDDLE``, ``RIGHT``, ``PRIMARY``, ``SECONDARY``, ``1``, ``2``, ``3``, ``4``,
    ``5``, ``6``, or ``7`` for the button argument and returns either ``LEFT``, ``MIDDLE``, ``RIGHT``, ``4``, ``5``,
    ``6``, or ``7``. The ``PRIMARY``, ``SECONDARY``, ``1``, ``2``, and ``3`` values are never returned.

    The ``'left'`` and ``'right'`` mouse buttons will always refer to the physical left and right
    buttons on the mouse. The same applies for buttons 1 and 3.

    However, if ``button`` is ``'primary'`` or ``'secondary'``, then we must check if
    the mouse buttons have been "swapped" (for left-handed users) by the operating system's mouse
    settings.

    If the buttons are swapped, the primary button is the right mouse button and the secondary button is the left mouse
    button. If not swapped, the primary and secondary buttons are the left and right buttons, respectively.

    NOTE: Swap detection has not been implemented yet.
    """
    button = button.lower()
    if platform.system() == 'Linux':
        if button not in (LEFT, MIDDLE, RIGHT, PRIMARY, SECONDARY, 1, 2, 3, 4, 5, 6, 7):
            raise PyAutoGUIException("button argument must be one of ('left', 'middle', 'right', 'primary', 'secondary', 1, 2, 3, 4, 5, 6, 7)")
    elif button not in (LEFT, MIDDLE, RIGHT, PRIMARY, SECONDARY, 1, 2, 3):
        raise PyAutoGUIException("button argument must be one of ('left', 'middle', 'right', 'primary', 'secondary', 1, 2, 3)")
    if button in (PRIMARY, SECONDARY):
        swapped = platformModule._mouse_is_swapped()
        if swapped:
            if button == PRIMARY:
                return RIGHT
            elif button == SECONDARY:
                return LEFT
        elif button == PRIMARY:
            return LEFT
        elif button == SECONDARY:
            return RIGHT
    return {LEFT: LEFT, MIDDLE: MIDDLE, RIGHT: RIGHT, 1: LEFT, 2: MIDDLE, 3: RIGHT, 4: 4, 5: 5, 6: 6, 7: 7}[button]