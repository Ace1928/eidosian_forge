from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def updateFilter(self, opts):
    self.setEnumVals(opts)