import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _maybeRebuildAtlas(self, threshold=4, minlen=1000):
    n = len(self.fragmentAtlas)
    if n > minlen and n > threshold * len(self.data):
        self.fragmentAtlas.rebuild(list(zip(*self._style(['symbol', 'size', 'pen', 'brush']))))
        self.data['sourceRect'] = 0
        self.updateSpots()