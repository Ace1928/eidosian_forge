from enum import Enum
from ...Qt import QT_LIB, QtCore
from .list import ListParameter
def _getAllowedEnums(self, enum):
    """Pyside provides a dict for easy evaluation"""
    if issubclass(enum, Enum):
        vals = {e.name: e for e in enum}
    elif 'PySide' in QT_LIB:
        vals = enum.values
    elif 'PyQt5' in QT_LIB:
        vals = {}
        for key in dir(self.searchObj):
            value = getattr(self.searchObj, key)
            if isinstance(value, enum):
                vals[key] = value
    else:
        raise RuntimeError(f'Cannot find associated enum values for qt lib {QT_LIB}')
    vals.pop(f'M{enum.__name__}', None)
    return vals