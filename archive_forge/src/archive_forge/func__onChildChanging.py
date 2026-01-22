from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def _onChildChanging(self, child, value):
    if self.opts['exclusive'] and value:
        value = self.forward[child.name()]
    else:
        value = self.childrenValue()
    self.sigValueChanging.emit(self, value)