from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _compute_polygon_vertex(centroid, polygon_radius, angle):
    start_point = [polygon_radius, 0]
    return _apply_rotation(start_point, angle, centroid)