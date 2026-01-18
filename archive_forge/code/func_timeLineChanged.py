import os
from math import log10
from time import perf_counter
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..graphicsItems.GradientEditorItem import addGradientListToDocstring
from ..graphicsItems.ImageItem import ImageItem
from ..graphicsItems.InfiniteLine import InfiniteLine
from ..graphicsItems.LinearRegionItem import LinearRegionItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.VTickGroup import VTickGroup
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
from . import ImageViewTemplate_generic as ui_template
def timeLineChanged(self):
    if not self.ignoreTimeLine:
        self.play(0)
    ind, time = self.timeIndex(self.timeLine)
    if ind != self.currentIndex:
        self.currentIndex = ind
        self.updateImage()
    if self.discreteTimeLine:
        with fn.SignalBlock(self.timeLine.sigPositionChanged, self.timeLineChanged):
            if self.tVals is not None:
                self.timeLine.setPos(self.tVals[ind])
            else:
                self.timeLine.setPos(ind)
    self.sigTimeChanged.emit(ind, time)