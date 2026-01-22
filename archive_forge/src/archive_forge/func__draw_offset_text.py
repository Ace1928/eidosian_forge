import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _draw_offset_text(self, renderer, edgep1, edgep2, labeldeltas, centers, highs, pep, dx, dy):
    info = self._axinfo
    index = info['i']
    juggled = info['juggled']
    tickdir = info['tickdir']
    if juggled[2] == 2:
        outeredgep = edgep1
        outerindex = 0
    else:
        outeredgep = edgep2
        outerindex = 1
    pos = _move_from_center(outeredgep, centers, labeldeltas, self._axmask())
    olx, oly, olz = proj3d.proj_transform(*pos, self.axes.M)
    self.offsetText.set_text(self.major.formatter.get_offset())
    self.offsetText.set_position((olx, oly))
    angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
    self.offsetText.set_rotation(angle)
    self.offsetText.set_rotation_mode('anchor')
    centpt = proj3d.proj_transform(*centers, self.axes.M)
    if centpt[tickdir] > pep[tickdir, outerindex]:
        if centpt[index] <= pep[index, outerindex] and np.count_nonzero(highs) % 2 == 0:
            if highs.tolist() == [False, True, True] and index in (1, 2):
                align = 'left'
            else:
                align = 'right'
        else:
            align = 'left'
    elif centpt[index] > pep[index, outerindex] and np.count_nonzero(highs) % 2 == 0:
        align = 'right' if index == 2 else 'left'
    else:
        align = 'right'
    self.offsetText.set_va('center')
    self.offsetText.set_ha(align)
    self.offsetText.draw(renderer)