from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
class RadioParameterItem(BoolParameterItem):
    """
    Allows radio buttons to function as booleans when `exclusive` is *True*
    """

    def __init__(self, param, depth):
        self.emitter = Emitter()
        super().__init__(param, depth)

    def makeWidget(self):
        w = QtWidgets.QRadioButton()
        w.value = w.isChecked
        w.setValue = w.setChecked
        w.sigChanged = self.emitter.sigChanged
        w.toggled.connect(self.maybeSigChanged)
        self.hideWidget = False
        return w

    def maybeSigChanged(self, val):
        """
        Make sure to only activate on a "true" value, since an exclusive button group fires once to deactivate
        the old option and once to activate the new selection
        """
        if not val:
            return
        self.emitter.sigChanged.emit(self, val)