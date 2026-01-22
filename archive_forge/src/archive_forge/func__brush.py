import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _brush(self, prop):
    brushstyle = prop.get('brushstyle')
    if brushstyle in ('LinearGradientPattern', 'ConicalGradientPattern', 'RadialGradientPattern'):
        gradient = self._gradient(prop[0])
        brush = self.factory.createQObject('QBrush', 'brush', (gradient,), is_attribute=False)
    else:
        color = self._color(prop[0])
        brush = self.factory.createQObject('QBrush', 'brush', (color,), is_attribute=False)
        brushstyle = getattr(QtCore.Qt, brushstyle)
        brush.setStyle(brushstyle)
    return brush