from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def tri_area(t1, t2, t3):
    t1x, t1y = t1[0:2]
    t2x, t2y = t2[0:2]
    t3x, t3y = t3[0:2]
    return abs(t1x * (t2y - t3y) + t2x * (t3y - t1y) + t3x * (t1y - t2y)) / 2