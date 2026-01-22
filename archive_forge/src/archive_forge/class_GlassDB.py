import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
class GlassDB:
    """
    Database of dispersion coefficients for Schott glasses
     + Corning 7980
    """

    def __init__(self, fileName='schott_glasses.csv'):
        path = os.path.dirname(__file__)
        fh = gzip.open(os.path.join(path, 'schott_glasses.csv.gz'), 'rb')
        r = csv.reader(map(str, fh.readlines()))
        lines = [x for x in r]
        self.data = {}
        header = lines[0]
        for l in lines[1:]:
            info = {}
            for i in range(1, len(l)):
                info[header[i]] = l[i]
            self.data[l[0]] = info
        self.data['Corning7980'] = {'B1': 0.683740494, 'B2': 0.420323613, 'B3': 0.58502748, 'C1': 0.00460352869, 'C2': 0.0133968856, 'C3': 64.4932732, 'TAUI25/250': 0.95, 'TAUI25/1400': 0.98}
        for k in self.data:
            self.data[k]['ior_cache'] = {}

    def ior(self, glass, wl):
        """
        Return the index of refraction for *glass* at wavelength *wl*.
        
        The *glass* argument must be a key in self.data.
        """
        info = self.data[glass]
        cache = info['ior_cache']
        if wl not in cache:
            B = list(map(float, [info['B1'], info['B2'], info['B3']]))
            C = list(map(float, [info['C1'], info['C2'], info['C3']]))
            w2 = (wl / 1000.0) ** 2
            n = sqrt(1.0 + B[0] * w2 / (w2 - C[0]) + B[1] * w2 / (w2 - C[1]) + B[2] * w2 / (w2 - C[2]))
            cache[wl] = n
        return cache[wl]

    def transmissionCurve(self, glass):
        data = self.data[glass]
        keys = [int(x[7:]) for x in data.keys() if 'TAUI25' in x]
        keys.sort()
        curve = np.empty((2, len(keys)))
        for i in range(len(keys)):
            curve[0][i] = keys[i]
            key = 'TAUI25/%d' % keys[i]
            val = data[key]
            if val == '':
                val = 0
            else:
                val = float(val)
            curve[1][i] = val
        return curve