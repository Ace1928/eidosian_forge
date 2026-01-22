import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class Mirror(Optic):

    def __init__(self, **params):
        defaults = {'r1': 0, 'r2': 0, 'd': 0.01}
        defaults.update(params)
        d = defaults.pop('d')
        defaults['x1'] = -d / 2.0
        defaults['x2'] = d / 2.0
        gitem = CircularSolid(brush=(100, 100, 100, 255), **defaults)
        Optic.__init__(self, gitem, **defaults)

    def propagateRay(self, ray):
        """Refract, reflect, absorb, and/or scatter ray. This function may create and return new rays"""
        surface = self.surfaces[0]
        p1, ai = surface.intersectRay(ray)
        if p1 is not None:
            p1 = surface.mapToItem(ray, p1)
            rd = ray['dir']
            a1 = atan2(rd[1], rd[0])
            ar = a1 + np.pi - 2 * ai
            ray.setEnd(p1)
            dp = Point(cos(ar), sin(ar))
            ray = Ray(parent=ray, dir=dp)
        else:
            ray.setEnd(None)
        return [ray]