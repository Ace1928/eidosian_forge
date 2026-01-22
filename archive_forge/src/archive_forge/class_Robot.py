from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
class Robot(RobotPart):

    def __init__(self):
        super(Robot, self).__init__()
        self.torsoItem = RobotTorso(self)
        self.headItem = RobotHead(self.torsoItem)
        self.upperLeftArmItem = RobotLimb(self.torsoItem)
        self.lowerLeftArmItem = RobotLimb(self.upperLeftArmItem)
        self.upperRightArmItem = RobotLimb(self.torsoItem)
        self.lowerRightArmItem = RobotLimb(self.upperRightArmItem)
        self.upperRightLegItem = RobotLimb(self.torsoItem)
        self.lowerRightLegItem = RobotLimb(self.upperRightLegItem)
        self.upperLeftLegItem = RobotLimb(self.torsoItem)
        self.lowerLeftLegItem = RobotLimb(self.upperLeftLegItem)
        self.timeline = QtCore.QTimeLine()
        settings = [(self.headItem, 0, -18, 20, -20), (self.upperLeftArmItem, -15, -10, 190, 180), (self.lowerLeftArmItem, 30, 0, 50, 10), (self.upperRightArmItem, 15, -10, 300, 310), (self.lowerRightArmItem, 30, 0, 0, -70), (self.upperRightLegItem, 10, 32, 40, 120), (self.lowerRightLegItem, 30, 0, 10, 50), (self.upperLeftLegItem, -10, 32, 150, 80), (self.lowerLeftLegItem, 30, 0, 70, 10), (self.torsoItem, 0, 0, 5, -20)]
        self.animations = []
        for item, pos_x, pos_y, rotation1, rotation2 in settings:
            item.setPos(pos_x, pos_y)
            animation = QtWidgets.QGraphicsItemAnimation()
            animation.setItem(item)
            animation.setTimeLine(self.timeline)
            animation.setRotationAt(0, rotation1)
            animation.setRotationAt(1, rotation2)
            self.animations.append(animation)
        self.animations[0].setScaleAt(1, 1.1, 1.1)
        self.timeline.setUpdateInterval(1000 / 25)
        self.timeline.setCurveShape(QtCore.QTimeLine.SineCurve)
        self.timeline.setLoopCount(0)
        self.timeline.setDuration(2000)
        self.timeline.start()

    def boundingRect(self):
        return QtCore.QRectF()

    def paint(self, painter, option, widget=None):
        pass