from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def updateCurrentTime(self, currentTime):
    if self.m_pathType == Animation.CirclePath:
        if self.m_path.isEmpty():
            end = self.endValue()
            start = self.startValue()
            self.m_path.moveTo(start)
            self.m_path.addEllipse(QtCore.QRectF(start, end))
        dura = self.duration()
        if dura == 0:
            progress = 1.0
        else:
            progress = ((currentTime - 1) % dura + 1) / float(dura)
        easedProgress = self.easingCurve().valueForProgress(progress)
        if easedProgress > 1.0:
            easedProgress -= 1.0
        elif easedProgress < 0:
            easedProgress += 1.0
        pt = self.m_path.pointAtPercent(easedProgress)
        self.updateCurrentValue(pt)
        self.valueChanged.emit(pt)
    else:
        super(Animation, self).updateCurrentTime(currentTime)