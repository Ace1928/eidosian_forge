from ... import functions as fn
from ...widgets.ColorButton import ColorButton
from .basetypes import SimpleParameter, WidgetParameterItem
class ColorParameter(SimpleParameter):
    itemClass = ColorParameterItem

    def _interpretValue(self, v):
        return fn.mkColor(v)

    def value(self):
        value = super().value()
        if value is None:
            return None
        return fn.mkColor(value)

    def saveState(self, filter=None):
        state = super().saveState(filter)
        state['value'] = self.value().getRgb()
        return state