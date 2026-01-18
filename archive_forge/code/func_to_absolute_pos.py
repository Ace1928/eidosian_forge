import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
def to_absolute_pos(self, nx, ny, x_max, y_max, rotation):
    """Transforms normalized (0-1) coordinates `nx` and `ny` to absolute
        coordinates using `x_max`, `y_max` and `rotation`.

        :raises:
            `ValueError`: If `rotation` is not one of: 0, 90, 180 or 270

        .. versionadded:: 2.1.0
        """
    if rotation == 0:
        return (nx * x_max, ny * y_max)
    elif rotation == 90:
        return (ny * y_max, (1 - nx) * x_max)
    elif rotation == 180:
        return ((1 - nx) * x_max, (1 - ny) * y_max)
    elif rotation == 270:
        return ((1 - ny) * y_max, nx * x_max)
    raise ValueError('Invalid rotation %s, valid values are 0, 90, 180 or 270' % rotation)