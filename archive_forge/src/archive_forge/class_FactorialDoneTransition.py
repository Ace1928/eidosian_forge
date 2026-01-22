from PySide2.QtWidgets import *
from PySide2.QtCore import *
class FactorialDoneTransition(QSignalTransition):

    def __init__(self, fact):
        super(FactorialDoneTransition, self).__init__(fact, SIGNAL('xChanged(int)'))
        self.fact = fact

    def eventTest(self, e):
        if not super(FactorialDoneTransition, self).eventTest(e):
            return False
        return e.arguments()[0] <= 1

    def onTransition(self, e):
        print(self.fact.fac)