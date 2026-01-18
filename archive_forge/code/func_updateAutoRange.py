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
def updateAutoRange(self):
    if self._updatingRange:
        return
    self._updatingRange = True
    try:
        if not any(self.state['autoRange']):
            return
        targetRect = self.viewRange()
        fractionVisible = self.state['autoRange'][:]
        for i in [0, 1]:
            if type(fractionVisible[i]) is bool:
                fractionVisible[i] = 1.0
        childRange = None
        order = [0, 1]
        if self.state['autoVisibleOnly'][0] is True:
            order = [1, 0]
        args = {}
        for ax in order:
            if self.state['autoRange'][ax] is False:
                continue
            if self.state['autoVisibleOnly'][ax]:
                oRange = [None, None]
                oRange[ax] = targetRect[1 - ax]
                childRange = self.childrenBounds(frac=fractionVisible, orthoRange=oRange)
            elif childRange is None:
                childRange = self.childrenBounds(frac=fractionVisible)
            xr = childRange[ax]
            if xr is not None:
                if self.state['autoPan'][ax]:
                    x = sum(xr) * 0.5
                    w2 = (targetRect[ax][1] - targetRect[ax][0]) / 2.0
                    childRange[ax] = [x - w2, x + w2]
                else:
                    padding = self.suggestPadding(ax)
                    wp = (xr[1] - xr[0]) * padding
                    childRange[ax][0] -= wp
                    childRange[ax][1] += wp
                targetRect[ax] = childRange[ax]
                args['xRange' if ax == 0 else 'yRange'] = targetRect[ax]
        for k in ['xRange', 'yRange']:
            if k in args:
                if not math.isfinite(args[k][0]) or not math.isfinite(args[k][1]):
                    _ = args.pop(k)
        if len(args) == 0:
            return
        args['padding'] = 0.0
        args['disableAutoRange'] = False
        self.setRange(**args)
    finally:
        self._autoRangeNeedsUpdate = False
        self._updatingRange = False