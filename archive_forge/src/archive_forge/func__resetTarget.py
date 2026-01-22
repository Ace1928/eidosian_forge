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
def _resetTarget(self, force: bool=False):
    if self.state['aspectLocked'] is False or force:
        self.state['targetRange'] = [self.state['viewRange'][0][:], self.state['viewRange'][1][:]]