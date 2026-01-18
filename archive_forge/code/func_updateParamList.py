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
def updateParamList(self):
    self.ctrl.avgParamList.clear()
    for c in self.curves:
        for p in list(self.itemMeta.get(c, {}).keys()):
            if type(p) is tuple:
                p = '.'.join(p)
            matches = self.ctrl.avgParamList.findItems(p, QtCore.Qt.MatchFlag.MatchExactly)
            if len(matches) == 0:
                i = QtWidgets.QListWidgetItem(p)
                if p in self.paramList and self.paramList[p] is True:
                    i.setCheckState(QtCore.Qt.CheckState.Checked)
                else:
                    i.setCheckState(QtCore.Qt.CheckState.Unchecked)
                self.ctrl.avgParamList.addItem(i)
            else:
                i = matches[0]
            self.paramList[p] = i.checkState() == QtCore.Qt.CheckState.Checked