import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class Lens(Optic):

    def __init__(self, **params):
        defaults = {'dia': 25.4, 'r1': 50.0, 'r2': 0, 'd': 4.0, 'glass': 'N-BK7', 'reflect': False}
        defaults.update(params)
        d = defaults.pop('d')
        defaults['x1'] = -d / 2.0
        defaults['x2'] = d / 2.0
        gitem = CircularSolid(brush=(100, 100, 130, 100), **defaults)
        Optic.__init__(self, gitem, **defaults)

    def propagateRay(self, ray):
        """Refract, reflect, absorb, and/or scatter ray. This function may create and return new rays"""
        '\n        NOTE:: We can probably use this to compute refractions faster: (from GLSL 120 docs)\n\n        For the incident vector I and surface normal N, and the\n        ratio of indices of refraction eta, return the refraction\n        vector. The result is computed by\n        k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I))\n        if (k < 0.0)\n            return genType(0.0)\n        else\n            return eta * I - (eta * dot(N, I) + sqrt(k)) * N\n        The input parameters for the incident vector I and the\n        surface normal N must already be normalized to get the\n        desired results. eta == ratio of IORs\n\n\n        For reflection:\n        For the incident vector I and surface orientation N,\n        returns the reflection direction:\n        I – 2 ∗ dot(N, I) ∗ N\n        N must already be normalized in order to achieve the\n        desired result.\n        '
        iors = [self.ior(ray['wl']), 1.0]
        for i in [0, 1]:
            surface = self.surfaces[i]
            ior = iors[i]
            p1, ai = surface.intersectRay(ray)
            if p1 is None:
                ray.setEnd(None)
                break
            p1 = surface.mapToItem(ray, p1)
            rd = ray['dir']
            a1 = atan2(rd[1], rd[0])
            try:
                ar = a1 - ai + asin(sin(ai) * ray['ior'] / ior)
            except ValueError:
                ar = np.nan
            ray.setEnd(p1)
            dp = Point(cos(ar), sin(ar))
            ray = Ray(parent=ray, ior=ior, dir=dp)
        return [ray]