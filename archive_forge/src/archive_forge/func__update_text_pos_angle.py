import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def _update_text_pos_angle(self, arrow):
    path_disp = self._get_arrow_path_disp(arrow)
    (x1, y1), (cx, cy), (x2, y2) = path_disp.vertices
    t = self.label_pos
    tt = 1 - t
    x = tt ** 2 * x1 + 2 * t * tt * cx + t ** 2 * x2
    y = tt ** 2 * y1 + 2 * t * tt * cy + t ** 2 * y2
    if self.labels_horizontal:
        angle = 0
    else:
        change_x = 2 * tt * (cx - x1) + 2 * t * (x2 - cx)
        change_y = 2 * tt * (cy - y1) + 2 * t * (y2 - cy)
        angle = np.arctan2(change_y, change_x) / (2 * np.pi) * 360
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
    x, y = self.ax.transData.inverted().transform((x, y))
    return (x, y, angle)