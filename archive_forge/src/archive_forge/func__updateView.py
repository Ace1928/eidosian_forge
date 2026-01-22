import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
def _updateView(self):
    if not hasattr(self, '_connectedView'):
        return
    self.forgetViewBox()
    self.forgetViewWidget()
    view = self.getViewBox()
    oldView = None
    if self._connectedView is not None:
        oldView = self._connectedView()
    if view is oldView:
        return
    if oldView is not None:
        Device = 'Device' if hasattr(oldView, 'sigDeviceRangeChanged') else ''
        for signal, slot in [(f'sig{Device}RangeChanged', self.viewRangeChanged), (f'sig{Device}TransformChanged', self.viewTransformChanged)]:
            try:
                getattr(oldView, signal).disconnect(slot)
            except (TypeError, AttributeError, RuntimeError):
                pass
        self._connectedView = None
    if view is not None:
        if hasattr(view, 'sigDeviceRangeChanged'):
            view.sigDeviceRangeChanged.connect(self.viewRangeChanged)
            view.sigDeviceTransformChanged.connect(self.viewTransformChanged)
        else:
            view.sigRangeChanged.connect(self.viewRangeChanged)
            view.sigTransformChanged.connect(self.viewTransformChanged)
        self._connectedView = weakref.ref(view)
        self.viewRangeChanged()
        self.viewTransformChanged()
    self._replaceView(oldView)
    self.viewChanged(view, oldView)