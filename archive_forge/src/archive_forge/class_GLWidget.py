import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
class GLWidget(QtWidgets.QOpenGLWidget):
    xRotationChanged = QtCore.Signal(int)
    yRotationChanged = QtCore.Signal(int)
    zRotationChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.object = 0
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.lastPos = QtCore.QPoint()
        self.trolltechGreen = QtGui.QColor.fromCmykF(0.4, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.emit(QtCore.SIGNAL('xRotationChanged(int)'), angle)
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.emit(QtCore.SIGNAL('yRotationChanged(int)'), angle)
            self.update()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.emit(QtCore.SIGNAL('zRotationChanged(int)'), angle)
            self.update()

    def initializeGL(self):
        darkTrolltechPurple = self.trolltechPurple.darker()
        GL.glClearColor(darkTrolltechPurple.redF(), darkTrolltechPurple.greenF(), darkTrolltechPurple.blueF(), darkTrolltechPurple.alphaF())
        self.object = self.makeObject()
        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslated(0.0, 0.0, -10.0)
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
        GL.glCallList(self.object)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport(int((width - side) / 2), int((height - side) / 2), side, side)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setZRotation(self.zRot + 8 * dx)
        self.lastPos = QtCore.QPoint(event.pos())

    def makeObject(self):
        genList = GL.glGenLists(1)
        GL.glNewList(genList, GL.GL_COMPILE)
        GL.glBegin(GL.GL_QUADS)
        x1 = +0.06
        y1 = -0.14
        x2 = +0.14
        y2 = -0.06
        x3 = +0.08
        y3 = +0.0
        x4 = +0.3
        y4 = +0.22
        self.quad(x1, y1, x2, y2, y2, x2, y1, x1)
        self.quad(x3, y3, x4, y4, y4, x4, y3, x3)
        self.extrude(x1, y1, x2, y2)
        self.extrude(x2, y2, y2, x2)
        self.extrude(y2, x2, y1, x1)
        self.extrude(y1, x1, x1, y1)
        self.extrude(x3, y3, x4, y4)
        self.extrude(x4, y4, y4, x4)
        self.extrude(y4, x4, y3, x3)
        Pi = 3.141592653589793
        NumSectors = 200
        for i in range(NumSectors):
            angle1 = i * 2 * Pi / NumSectors
            x5 = 0.3 * math.sin(angle1)
            y5 = 0.3 * math.cos(angle1)
            x6 = 0.2 * math.sin(angle1)
            y6 = 0.2 * math.cos(angle1)
            angle2 = (i + 1) * 2 * Pi / NumSectors
            x7 = 0.2 * math.sin(angle2)
            y7 = 0.2 * math.cos(angle2)
            x8 = 0.3 * math.sin(angle2)
            y8 = 0.3 * math.cos(angle2)
            self.quad(x5, y5, x6, y6, x7, y7, x8, y8)
            self.extrude(x6, y6, x7, y7)
            self.extrude(x8, y8, x5, y5)
        GL.glEnd()
        GL.glEndList()
        return genList

    def quad(self, x1, y1, x2, y2, x3, y3, x4, y4):
        GL.glColor(self.trolltechGreen.redF(), self.trolltechGreen.greenF(), self.trolltechGreen.blueF(), self.trolltechGreen.alphaF())
        GL.glVertex3d(x1, y1, +0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x3, y3, +0.05)
        GL.glVertex3d(x4, y4, +0.05)
        GL.glVertex3d(x4, y4, -0.05)
        GL.glVertex3d(x3, y3, -0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x1, y1, -0.05)

    def extrude(self, x1, y1, x2, y2):
        darkTrolltechGreen = self.trolltechGreen.darker(250 + int(100 * x1))
        GL.glColor(darkTrolltechGreen.redF(), darkTrolltechGreen.greenF(), darkTrolltechGreen.blueF(), darkTrolltechGreen.alphaF())
        GL.glVertex3d(x1, y1, -0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x1, y1, +0.05)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def freeResources(self):
        self.makeCurrent()
        GL.glDeleteLists(self.object, 1)