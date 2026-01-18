import sys
import math
import numpy
import ctypes
from PySide2.QtCore import QCoreApplication, Signal, SIGNAL, SLOT, Qt, QSize, QPoint
from PySide2.QtGui import (QVector3D, QOpenGLFunctions, QOpenGLVertexArrayObject, QOpenGLBuffer,
from PySide2.QtWidgets import (QApplication, QWidget, QMessageBox, QHBoxLayout, QSlider,
from shiboken2 import VoidPtr
def vertexShaderSourceCore(self):
    return '#version 150\n                in vec4 vertex;\n                in vec3 normal;\n                out vec3 vert;\n                out vec3 vertNormal;\n                uniform mat4 projMatrix;\n                uniform mat4 mvMatrix;\n                uniform mat3 normalMatrix;\n                void main() {\n                   vert = vertex.xyz;\n                   vertNormal = normalMatrix * normal;\n                   gl_Position = projMatrix * mvMatrix * vertex;\n                }'