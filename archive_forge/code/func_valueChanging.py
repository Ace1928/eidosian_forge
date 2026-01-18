import ast
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
def valueChanging(sb, value):
    changingLabel.setText('Value changing: %s' % str(sb.value()))