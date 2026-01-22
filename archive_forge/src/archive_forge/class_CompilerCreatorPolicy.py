import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
class CompilerCreatorPolicy(object):

    def __init__(self, all_pyside2_modules):
        self._modules = []
        pyside2_modules[:] = all_pyside2_modules

    def createQtGuiWrapper(self):
        return _QtGuiWrapper

    def createQtWidgetsWrapper(self):
        return _QtWidgetsWrapper

    def createModuleWrapper(self, name, classes):
        mw = _ModuleWrapper(name, classes)
        self._modules.append(mw)
        return mw

    def createCustomWidgetLoader(self):
        cw = _CustomWidgetLoader()
        self._modules.append(cw)
        return cw

    def instantiate(self, clsObject, objectname, ctor_args, is_attribute=True, no_instantiation=False):
        return clsObject(objectname, is_attribute, ctor_args, no_instantiation)

    def invoke(self, rname, method, args):
        return method(rname, *args)

    def getSlot(self, object, slotname):
        return Literal('%s.%s' % (object, slotname))

    def _writeOutImports(self):
        for module in self._modules:
            module._writeImportCode()