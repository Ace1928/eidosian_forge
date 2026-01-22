import sys
from PyQt5 import QtGui, QtWidgets
class LoaderCreatorPolicy(object):

    def __init__(self, package):
        self._package = package

    def createQtGuiWidgetsWrappers(self):
        return [_QtGuiWrapper, _QtWidgetsWrapper]

    def createModuleWrapper(self, moduleName, classes):
        return _ModuleWrapper(moduleName, classes)

    def createCustomWidgetLoader(self):
        return _CustomWidgetLoader(self._package)

    def instantiate(self, clsObject, objectName, ctor_args, is_attribute=True):
        return clsObject(*ctor_args)

    def invoke(self, rname, method, args):
        return method(*args)

    def getSlot(self, object, slotname):
        if slotname == 'raise':
            slotname += '_'
        return getattr(object, slotname)

    def asString(self, s):
        return s