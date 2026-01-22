import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _gradient(self, prop):
    name = 'gradient'
    gtype = prop.get('type', '')
    if gtype == 'LinearGradient':
        startx = float(prop.get('startx'))
        starty = float(prop.get('starty'))
        endx = float(prop.get('endx'))
        endy = float(prop.get('endy'))
        gradient = self.factory.createQObject('QLinearGradient', name, (startx, starty, endx, endy), is_attribute=False)
    elif gtype == 'ConicalGradient':
        centralx = float(prop.get('centralx'))
        centraly = float(prop.get('centraly'))
        angle = float(prop.get('angle'))
        gradient = self.factory.createQObject('QConicalGradient', name, (centralx, centraly, angle), is_attribute=False)
    elif gtype == 'RadialGradient':
        centralx = float(prop.get('centralx'))
        centraly = float(prop.get('centraly'))
        radius = float(prop.get('radius'))
        focalx = float(prop.get('focalx'))
        focaly = float(prop.get('focaly'))
        gradient = self.factory.createQObject('QRadialGradient', name, (centralx, centraly, radius, focalx, focaly), is_attribute=False)
    else:
        raise UnsupportedPropertyError(prop.tag)
    spread = prop.get('spread')
    if spread:
        gradient.setSpread(getattr(QtGui.QGradient, spread))
    cmode = prop.get('coordinatemode')
    if cmode:
        gradient.setCoordinateMode(getattr(QtGui.QGradient, cmode))
    for gstop in prop:
        if gstop.tag != 'gradientstop':
            raise UnsupportedPropertyError(gstop.tag)
        position = float(gstop.get('position'))
        color = self._color(gstop[0])
        gradient.setColorAt(position, color)
    return gradient