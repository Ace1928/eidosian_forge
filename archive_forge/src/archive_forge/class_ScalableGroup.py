from _buildParamTypes import makeAllParamTypes
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
class ScalableGroup(pTypes.GroupParameter):

    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = 'Add'
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, typ):
        val = {'str': '', 'float': 0.0, 'int': 0}[typ]
        self.addChild(dict(name='ScalableParam %d' % (len(self.childs) + 1), type=typ, value=val, removable=True, renamable=True))