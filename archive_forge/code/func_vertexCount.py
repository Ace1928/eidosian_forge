import sys
import math
import numpy
import ctypes
from PySide2.QtCore import QCoreApplication, Signal, SIGNAL, SLOT, Qt, QSize, QPoint
from PySide2.QtGui import (QVector3D, QOpenGLFunctions, QOpenGLVertexArrayObject, QOpenGLBuffer,
from PySide2.QtWidgets import (QApplication, QWidget, QMessageBox, QHBoxLayout, QSlider,
from shiboken2 import VoidPtr
def vertexCount(self):
    return self.m_count / 6