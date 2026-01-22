from ...colormap import ColorMap
from ...Qt import QtCore
from ...widgets.GradientWidget import GradientWidget
from .basetypes import SimpleParameter, WidgetParameterItem
class ColorMapParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a :class:`GradientWidget <pyqtgraph.GradientWidget>`"""

    def makeWidget(self):
        w = GradientWidget(orientation='bottom')
        w.sizeHint = lambda: QtCore.QSize(300, 35)
        w.sigChanged = w.sigGradientChangeFinished
        w.sigChanging = w.sigGradientChanged
        w.value = w.colorMap
        w.setValue = w.setColorMap
        self.hideWidget = False
        self.asSubItem = True
        return w