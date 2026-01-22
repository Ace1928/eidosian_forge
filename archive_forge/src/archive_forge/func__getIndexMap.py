from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import parametertree as ptree
from ..graphicsItems.TextItem import TextItem
from ..Qt import QtCore, QtWidgets
from .ColorMapWidget import ColorMapParameter
from .DataFilterWidget import DataFilterParameter
from .PlotWidget import PlotWidget
def _getIndexMap(self):
    if self._indexMap is None:
        self._indexMap = {j: i for i, j in enumerate(self._visibleIndices)}
    return self._indexMap