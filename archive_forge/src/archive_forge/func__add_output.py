import logging
from types import SimpleNamespace
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib import _docstring
def _add_output(self, path, angle, flow, length):
    """
        Append an output to a path and return its tip and label locations.

        .. note:: *flow* is negative for an output.
        """
    if angle is None:
        return ([0, 0], [0, 0])
    else:
        x, y = path[-1][1]
        tipheight = (self.shoulder - flow / 2) * self.pitch
        if angle == RIGHT:
            x += length
            tip = [x + tipheight, y + flow / 2.0]
            path.extend([(Path.LINETO, [x, y]), (Path.LINETO, [x, y + self.shoulder]), (Path.LINETO, tip), (Path.LINETO, [x, y - self.shoulder + flow]), (Path.LINETO, [x, y + flow]), (Path.LINETO, [x - self.gap, y + flow])])
            label_location = [tip[0] + self.offset, tip[1]]
        else:
            x += self.gap
            if angle == UP:
                sign, quadrant = (1, 3)
            else:
                sign, quadrant = (-1, 0)
            tip = [x - flow / 2.0, y + sign * (length + tipheight)]
            if self.radius:
                path.extend(self._arc(quadrant=quadrant, cw=angle == UP, radius=self.radius, center=(x - self.radius, y + sign * self.radius)))
            else:
                path.append((Path.LINETO, [x, y]))
            path.extend([(Path.LINETO, [x, y + sign * length]), (Path.LINETO, [x - self.shoulder, y + sign * length]), (Path.LINETO, tip), (Path.LINETO, [x + self.shoulder - flow, y + sign * length]), (Path.LINETO, [x - flow, y + sign * length])])
            path.extend(self._arc(quadrant=quadrant, cw=angle == DOWN, radius=self.radius - flow, center=(x - self.radius, y + sign * self.radius)))
            path.append((Path.LINETO, [x - flow, y + sign * flow]))
            label_location = [tip[0], tip[1] + sign * self.offset]
        return (tip, label_location)