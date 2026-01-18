import math
from typing import NamedTuple
from dataclasses import dataclass
def transformVector(self, v):
    """Transform an (dx, dy) vector, treating translation as zero.

        :Example:

                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVector((3, -4))
                (6, -8)
                >>>
        """
    dx, dy = v
    xx, xy, yx, yy = self[:4]
    return (xx * dx + yx * dy, xy * dx + yy * dy)