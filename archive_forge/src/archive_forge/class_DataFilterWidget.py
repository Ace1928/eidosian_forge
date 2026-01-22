from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class DataFilterWidget(ptree.ParameterTree):
    """
    This class allows the user to filter multi-column data sets by specifying
    multiple criteria
    
    Wraps methods from DataFilterParameter: setFields, generateMask,
    filterData, and describe.
    """
    sigFilterChanged = QtCore.Signal(object)

    def __init__(self):
        ptree.ParameterTree.__init__(self, showHeader=False)
        self.params = DataFilterParameter()
        self.setParameters(self.params)
        self.params.sigFilterChanged.connect(self.sigFilterChanged)
        self.setFields = self.params.setFields
        self.generateMask = self.params.generateMask
        self.filterData = self.params.filterData
        self.describe = self.params.describe

    def parameters(self):
        return self.params

    def addFilter(self, name):
        """Add a new filter and return the created parameter item.
        """
        return self.params.addNew(name)