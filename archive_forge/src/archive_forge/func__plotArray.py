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
def _plotArray(self, arr, x=None, **kargs):
    if arr.ndim != 1:
        raise Exception('Array must be 1D to plot (shape is %s)' % arr.shape)
    if x is None:
        x = np.arange(arr.shape[0])
    if x.ndim != 1:
        raise Exception('X array must be 1D to plot (shape is %s)' % x.shape)
    c = PlotCurveItem(arr, x=x, **kargs)
    return c