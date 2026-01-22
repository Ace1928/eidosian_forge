from PyQt5 import QtCore, QtGui, QtWidgets
from ..uiparser import UIParser
from .qobjectcreator import LoaderCreatorPolicy
class DynamicUILoader(UIParser):

    def __init__(self, package):
        UIParser.__init__(self, QtCore, QtGui, QtWidgets, LoaderCreatorPolicy(package))

    def createToplevelWidget(self, classname, widgetname):
        if self.toplevelInst is None:
            return self.factory.createQObject(classname, widgetname, ())
        if not isinstance(self.toplevelInst, self.factory.findQObjectType(classname)):
            raise TypeError(('Wrong base class of toplevel widget', (type(self.toplevelInst), classname)))
        return self.toplevelInst

    def loadUi(self, filename, toplevelInst, resource_suffix):
        self.toplevelInst = toplevelInst
        return self.parse(filename, resource_suffix)