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
@_genericPyAutoGUIChecks
def tripleClick(x=None, y=None, interval=0.0, button=LEFT, duration=0.0, tween=linear, logScreenshot=None, _pause=True):
    """Performs a triple click.

    This is a wrapper function for click('left', x, y, 3, interval).

    The x and y parameters detail where the mouse event happens. If None, the
    current mouse position is used. If a float value, it is rounded down. If
    outside the boundaries of the screen, the event happens at edge of the
    screen.

    Args:
      x (int, float, None, tuple, optional): The x position on the screen where the
        click happens. None by default. If tuple, this is used for x and y.
        If x is a str, it's considered a filename of an image to find on
        the screen with locateOnScreen() and click the center of.
      y (int, float, None, optional): The y position on the screen where the
        click happens. None by default.
      interval (float, optional): The number of seconds in between each click,
        if the number of clicks is greater than 1. 0.0 by default, for no
        pause in between clicks.
      button (str, int, optional): The mouse button released. TODO

    Returns:
      None

    Raises:
      PyAutoGUIException: If button is not one of 'left', 'middle', 'right', 1, 2, 3, 4,
        5, 6, or 7
    """
    if sys.platform == 'darwin':
        x, y = _normalizeXYArgs(x, y)
        _mouseMoveDrag('move', x, y, 0, 0, duration=0, tween=None)
        x, y = platformModule._position()
        _logScreenshot(logScreenshot, 'click', '%s,%s,%s,3' % (x, y, button), folder='.')
        platformModule._multiClick(x, y, button, 3)
    else:
        click(x, y, 3, interval, button, duration, tween, logScreenshot, _pause=False)