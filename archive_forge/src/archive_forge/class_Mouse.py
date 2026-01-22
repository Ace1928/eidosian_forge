import math
from PySide2 import QtCore, QtGui, QtWidgets
import mice_rc
class Mouse(QtWidgets.QGraphicsItem):
    Pi = math.pi
    TwoPi = 2.0 * Pi
    adjust = 0.5
    BoundingRect = QtCore.QRectF(-20 - adjust, -22 - adjust, 40 + adjust, 83 + adjust)

    def __init__(self):
        super(Mouse, self).__init__()
        self.angle = 0.0
        self.speed = 0.0
        self.mouseEyeDirection = 0.0
        self.color = QtGui.QColor(QtCore.qrand() % 256, QtCore.qrand() % 256, QtCore.qrand() % 256)
        self.setTransform(QtGui.QTransform().rotate(QtCore.qrand() % (360 * 16)))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(1000 / 33)

    @staticmethod
    def normalizeAngle(angle):
        while angle < 0:
            angle += Mouse.TwoPi
        while angle > Mouse.TwoPi:
            angle -= Mouse.TwoPi
        return angle

    def boundingRect(self):
        return Mouse.BoundingRect

    def shape(self):
        path = QtGui.QPainterPath()
        path.addRect(-10, -20, 20, 40)
        return path

    def paint(self, painter, option, widget):
        painter.setBrush(self.color)
        painter.drawEllipse(-10, -20, 20, 40)
        painter.setBrush(QtCore.Qt.white)
        painter.drawEllipse(-10, -17, 8, 8)
        painter.drawEllipse(2, -17, 8, 8)
        painter.setBrush(QtCore.Qt.black)
        painter.drawEllipse(QtCore.QRectF(-2, -22, 4, 4))
        painter.drawEllipse(QtCore.QRectF(-8.0 + self.mouseEyeDirection, -17, 4, 4))
        painter.drawEllipse(QtCore.QRectF(4.0 + self.mouseEyeDirection, -17, 4, 4))
        if self.scene().collidingItems(self):
            painter.setBrush(QtCore.Qt.red)
        else:
            painter.setBrush(QtCore.Qt.darkYellow)
        painter.drawEllipse(-17, -12, 16, 16)
        painter.drawEllipse(1, -12, 16, 16)
        path = QtGui.QPainterPath(QtCore.QPointF(0, 20))
        path.cubicTo(-5, 22, -5, 22, 0, 25)
        path.cubicTo(5, 27, 5, 32, 0, 30)
        path.cubicTo(-5, 32, -5, 42, 0, 35)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

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