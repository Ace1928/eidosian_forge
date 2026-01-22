import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def _text_shift(self):
    return {'N': (0, +self.labelsep), 'S': (0, -self.labelsep), 'E': (+self.labelsep, 0), 'W': (-self.labelsep, 0)}[self.labelpos]