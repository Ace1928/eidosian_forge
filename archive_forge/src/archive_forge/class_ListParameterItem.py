import warnings
from collections import OrderedDict
from ... import functions as fn
from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class ListParameterItem(WidgetParameterItem):
    """
    WidgetParameterItem subclass providing comboBox that lets the user select from a list of options.

    """

    def __init__(self, param, depth):
        self.targetValue = None
        WidgetParameterItem.__init__(self, param, depth)

    def makeWidget(self):
        w = QtWidgets.QComboBox()
        w.setMaximumHeight(20)
        w.sigChanged = w.currentIndexChanged
        w.value = self.value
        w.setValue = self.setValue
        self.widget = w
        self.limitsChanged(self.param, self.param.opts['limits'])
        if len(self.forward) > 0 and self.param.hasValue():
            self.setValue(self.param.value())
        return w

    def value(self):
        key = self.widget.currentText()
        return self.forward.get(key, None)

    def setValue(self, val):
        self.targetValue = val
        match = [fn.eq(val, limVal) for limVal in self.reverse[0]]
        if not any(match):
            self.widget.setCurrentIndex(0)
        else:
            idx = match.index(True)
            key = self.reverse[1][idx]
            ind = self.widget.findText(key)
            self.widget.setCurrentIndex(ind)

    def limitsChanged(self, param, limits):
        if len(limits) == 0:
            limits = ['']
        self.forward, self.reverse = ListParameter.mapping(limits)
        try:
            self.widget.blockSignals(True)
            val = self.targetValue
            self.widget.clear()
            for k in self.forward:
                self.widget.addItem(k)
                if k == val:
                    self.widget.setCurrentIndex(self.widget.count() - 1)
                    self.updateDisplayLabel()
        finally:
            self.widget.blockSignals(False)

    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.widget.currentText()
        super().updateDisplayLabel(value)