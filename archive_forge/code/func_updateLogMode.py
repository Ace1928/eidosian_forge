import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def updateLogMode(self):
    x = self.ctrl.logXCheck.isChecked()
    y = self.ctrl.logYCheck.isChecked()
    for i in self.items:
        if hasattr(i, 'setLogMode'):
            i.setLogMode(x, y)
    self.getAxis('bottom').setLogMode(x, y)
    self.getAxis('top').setLogMode(x, y)
    self.getAxis('left').setLogMode(x, y)
    self.getAxis('right').setLogMode(x, y)
    self.enableAutoRange()
    self.recomputeAverages()