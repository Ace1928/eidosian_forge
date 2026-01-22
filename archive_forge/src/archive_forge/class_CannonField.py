import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
class CannonField(QtWidgets.QWidget):
    angleChanged = QtCore.Signal(int)
    forceChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.currentAngle = 45
        self.currentForce = 0
        self.timerCount = 0
        self.autoShootTimer = QtCore.QTimer(self)
        self.connect(self.autoShootTimer, QtCore.SIGNAL('timeout()'), self.moveShot)
        self.shootAngle = 0
        self.shootForce = 0
        self.setPalette(QtGui.QPalette(QtGui.QColor(250, 250, 200)))
        self.setAutoFillBackground(True)

    def angle(self):
        return self.currentAngle

    @QtCore.Slot(int)
    def setAngle(self, angle):
        if angle < 5:
            angle = 5
        if angle > 70:
            angle = 70
        if self.currentAngle == angle:
            return
        self.currentAngle = angle
        self.update()
        self.emit(QtCore.SIGNAL('angleChanged(int)'), self.currentAngle)

    def force(self):
        return self.currentForce

    @QtCore.Slot(int)
    def setForce(self, force):
        if force < 0:
            force = 0
        if self.currentForce == force:
            return
        self.currentForce = force
        self.emit(QtCore.SIGNAL('forceChanged(int)'), self.currentForce)

    @QtCore.Slot()
    def shoot(self):
        if self.autoShootTimer.isActive():
            return
        self.timerCount = 0
        self.shootAngle = self.currentAngle
        self.shootForce = self.currentForce
        self.autoShootTimer.start(5)

    @QtCore.Slot()
    def moveShot(self):
        region = QtGui.QRegion(self.shotRect())
        self.timerCount += 1
        shotR = self.shotRect()
        if shotR.x() > self.width() or shotR.y() > self.height():
            self.autoShootTimer.stop()
        else:
            region = region.united(QtGui.QRegion(shotR))
        self.update(region)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paintCannon(painter)
        if self.autoShootTimer.isActive():
            self.paintShot(painter)

    def paintShot(self, painter):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.black)
        painter.drawRect(self.shotRect())
    barrelRect = QtCore.QRect(33, -4, 15, 8)

    def paintCannon(self, painter):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.blue)
        painter.save()
        painter.translate(0, self.height())
        painter.drawPie(QtCore.QRect(-35, -35, 70, 70), 0, 90 * 16)
        painter.rotate(-self.currentAngle)
        painter.drawRect(CannonField.barrelRect)
        painter.restore()

    def cannonRect(self):
        result = QtCore.QRect(0, 0, 50, 50)
        result.moveBottomLeft(self.rect().bottomLect())
        return result

    def shotRect(self):
        gravity = 4.0
        time = self.timerCount / 40.0
        velocity = self.shootForce
        radians = self.shootAngle * 3.14159265 / 180
        velx = velocity * math.cos(radians)
        vely = velocity * math.sin(radians)
        x0 = (CannonField.barrelRect.right() + 5) * math.cos(radians)
        y0 = (CannonField.barrelRect.right() + 5) * math.sin(radians)
        x = x0 + velx * time
        y = y0 + vely * time - 0.5 * gravity * time * time
        result = QtCore.QRect(0, 0, 6, 6)
        result.moveCenter(QtCore.QPoint(round(x), self.height() - 1 - round(y)))
        return result