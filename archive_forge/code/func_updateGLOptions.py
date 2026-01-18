from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def updateGLOptions(self, opts):
    """
        Modify the OpenGL state options to use immediately before drawing this item.
        *opts* must be a dictionary as specified by setGLOptions.
        Values may also be None, in which case the key will be ignored.
        """
    self.__glOpts.update(opts)