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
def updateGrid(self, *args):
    alpha = self.ctrl.gridAlphaSlider.value()
    x = alpha if self.ctrl.xGridCheck.isChecked() else False
    y = alpha if self.ctrl.yGridCheck.isChecked() else False
    self.getAxis('top').setGrid(x)
    self.getAxis('bottom').setGrid(x)
    self.getAxis('left').setGrid(y)
    self.getAxis('right').setGrid(y)