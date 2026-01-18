from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def useOpenGL(self, b=True):
    if b:
        HAVE_OPENGL = hasattr(QtWidgets, 'QOpenGLWidget')
        if not HAVE_OPENGL:
            raise Exception('Requested to use OpenGL with QGraphicsView, but QOpenGLWidget is not available.')
        v = QtWidgets.QOpenGLWidget()
    else:
        v = QtWidgets.QWidget()
    self.setViewport(v)