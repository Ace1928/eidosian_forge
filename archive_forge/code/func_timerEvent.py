import math
from PySide2 import QtCore, QtGui, QtWidgets
import mice_rc
def timerEvent(self):
    lineToCenter = QtCore.QLineF(QtCore.QPointF(0, 0), self.mapFromScene(0, 0))
    if lineToCenter.length() > 150:
        angleToCenter = math.acos(lineToCenter.dx() / lineToCenter.length())
        if lineToCenter.dy() < 0:
            angleToCenter = Mouse.TwoPi - angleToCenter
        angleToCenter = Mouse.normalizeAngle(Mouse.Pi - angleToCenter + Mouse.Pi / 2)
        if angleToCenter < Mouse.Pi and angleToCenter > Mouse.Pi / 4:
            self.angle += [-0.25, 0.25][self.angle < -Mouse.Pi / 2]
        elif angleToCenter >= Mouse.Pi and angleToCenter < Mouse.Pi + Mouse.Pi / 2 + Mouse.Pi / 4:
            self.angle += [-0.25, 0.25][self.angle < Mouse.Pi / 2]
    elif math.sin(self.angle) < 0:
        self.angle += 0.25
    elif math.sin(self.angle) > 0:
        self.angle -= 0.25
    dangerMice = self.scene().items(QtGui.QPolygonF([self.mapToScene(0, 0), self.mapToScene(-30, -50), self.mapToScene(30, -50)]))
    for item in dangerMice:
        if item is self:
            continue
        lineToMouse = QtCore.QLineF(QtCore.QPointF(0, 0), self.mapFromItem(item, 0, 0))
        angleToMouse = math.acos(lineToMouse.dx() / lineToMouse.length())
        if lineToMouse.dy() < 0:
            angleToMouse = Mouse.TwoPi - angleToMouse
        angleToMouse = Mouse.normalizeAngle(Mouse.Pi - angleToMouse + Mouse.Pi / 2)
        if angleToMouse >= 0 and angleToMouse < Mouse.Pi / 2:
            self.angle += 0.5
        elif angleToMouse <= Mouse.TwoPi and angleToMouse > Mouse.TwoPi - Mouse.Pi / 2:
            self.angle -= 0.5
    if len(dangerMice) > 1 and QtCore.qrand() % 10 == 0:
        if QtCore.qrand() % 1:
            self.angle += QtCore.qrand() % 100 / 500.0
        else:
            self.angle -= QtCore.qrand() % 100 / 500.0
    self.speed += (-50 + QtCore.qrand() % 100) / 100.0
    dx = math.sin(self.angle) * 10
    self.mouseEyeDirection = [dx / 5, 0.0][QtCore.qAbs(dx / 5) < 1]
    self.setTransform(QtGui.QTransform().rotate(dx))
    self.setPos(self.mapToParent(0, -(3 + math.sin(self.speed) * 3)))