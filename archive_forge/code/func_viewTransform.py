from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def viewTransform(self):
    """Return the transform mapping this item's local coordinate system to the 
        view coordinate system."""
    tr = self.__transform
    p = self
    while True:
        p = p.parentItem()
        if p is None:
            break
        tr = p.transform() * tr
    return Transform3D(tr)