import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def _effectiveLimits(self):
    if self.state['logMode'][0]:
        xlimits = (max(self.state['limits']['xLimits'][0], -307.6), min(self.state['limits']['xLimits'][1], +308.2))
    else:
        xlimits = self.state['limits']['xLimits']
    if self.state['logMode'][1]:
        ylimits = (max(self.state['limits']['yLimits'][0], -307.6), min(self.state['limits']['yLimits'][1], +308.2))
    else:
        ylimits = self.state['limits']['yLimits']
    return (xlimits, ylimits)