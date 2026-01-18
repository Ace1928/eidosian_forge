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
def viewGeometry(self):
    """Return the screen geometry of the viewbox"""
    v = self.scene().views()[0]
    b = self.vb.mapRectToScene(self.vb.boundingRect())
    wr = v.mapFromScene(b).boundingRect()
    pos = v.mapToGlobal(v.pos())
    wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
    return wr