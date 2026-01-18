from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def valueChanged(self, param, val):
    self.updateDefaultBtn()