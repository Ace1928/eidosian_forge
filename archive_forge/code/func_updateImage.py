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
def updateImage(self, autoHistogramRange=True):
    if self.image is None:
        return
    image = self.getProcessedImage()
    if autoHistogramRange:
        self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)
    if self.imageItem.axisOrder == 'col-major':
        axorder = ['t', 'x', 'y', 'c']
    else:
        axorder = ['t', 'y', 'x', 'c']
    axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
    image = image.transpose(axorder)
    if self.axes['t'] is not None:
        self.ui.roiPlot.show()
        image = image[self.currentIndex]
    self.imageItem.updateImage(image)