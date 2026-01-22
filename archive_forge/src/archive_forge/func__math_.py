from math import atan2, degrees, hypot
from .Qt import QtCore
def _math_(self, op, x):
    if not isinstance(x, QtCore.QPointF):
        x = Point(x)
    return Point(getattr(self.x(), op)(x.x()), getattr(self.y(), op)(x.y()))