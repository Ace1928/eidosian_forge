from PySide2.QtWidgets import *
from PySide2.QtCore import *
class Pinger(QState):

    def __init__(self, parent):
        super(Pinger, self).__init__(parent)

    def onEntry(self, e):
        self.p = PingEvent()
        self.machine().postEvent(self.p)
        print('ping?')