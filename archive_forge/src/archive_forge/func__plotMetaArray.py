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
def _plotMetaArray(self, arr, x=None, autoLabel=True, **kargs):
    if arr.ndim != 1:
        raise Exception('can only automatically plot 1 dimensional arrays.')
    try:
        xv = arr.xvals(0)
    except:
        if x is None:
            xv = np.arange(arr.shape[0])
        else:
            xv = x
    c = PlotCurveItem(**kargs)
    c.setData(x=xv, y=arr.view(np.ndarray))
    if autoLabel:
        name = arr._info[0].get('name', None)
        units = arr._info[0].get('units', None)
        self.setLabel('bottom', text=name, units=units)
        name = arr._info[1].get('name', None)
        units = arr._info[1].get('units', None)
        self.setLabel('left', text=name, units=units)
    return c