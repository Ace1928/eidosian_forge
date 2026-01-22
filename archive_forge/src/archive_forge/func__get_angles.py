from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _get_angles(n_sides, rotation):
    angles = []
    degrees = 360 / n_sides
    current_angle = 270 - 0.5 * degrees + rotation
    for _ in range(0, n_sides):
        angles.append(current_angle)
        current_angle += degrees
        if current_angle > 360:
            current_angle -= 360
    return angles