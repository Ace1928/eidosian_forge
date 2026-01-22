from .. import PlotItem
from .. import functions as fn
from ..Qt import QtCore, QtWidgets
from .Exporter import Exporter
class MatplotlibWindow(QtWidgets.QMainWindow):

    def __init__(self):
        from ..widgets import MatplotlibWidget
        QtWidgets.QMainWindow.__init__(self)
        self.mpl = MatplotlibWidget.MatplotlibWidget()
        self.setCentralWidget(self.mpl)
        self.show()

    def __getattr__(self, attr):
        return getattr(self.mpl, attr)

    def closeEvent(self, ev):
        MatplotlibExporter.windows.remove(self)
        self.deleteLater()