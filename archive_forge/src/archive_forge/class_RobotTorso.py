from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
class RobotTorso(RobotPart):

    def boundingRect(self):
        return QtCore.QRectF(-30, -20, 60, 60)

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.dragOver and self.color.lighter(130) or self.color)
        painter.drawRoundedRect(-20, -20, 40, 60, 25, 25, QtCore.Qt.RelativeSize)
        painter.drawEllipse(-25, -20, 20, 20)
        painter.drawEllipse(5, -20, 20, 20)
        painter.drawEllipse(-20, 22, 20, 20)
        painter.drawEllipse(0, 22, 20, 20)