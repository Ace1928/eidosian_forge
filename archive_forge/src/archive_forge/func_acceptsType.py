import inspect
import weakref
from .Qt import QtCore, QtWidgets
def acceptsType(self, obj):
    for c in WidgetGroup.classes:
        if isinstance(obj, c):
            return True
    if hasattr(obj, 'widgetGroupInterface'):
        return True
    return False