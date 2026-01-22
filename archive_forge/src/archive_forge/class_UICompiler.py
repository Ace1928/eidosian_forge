import sys
from pyside2uic.properties import Properties
from pyside2uic.uiparser import UIParser
from pyside2uic.Compiler import qtproxies
from pyside2uic.Compiler.indenter import createCodeIndenter, getIndenter, \
from pyside2uic.Compiler.qobjectcreator import CompilerCreatorPolicy
from pyside2uic.Compiler.misc import write_import
class UICompiler(UIParser):

    def __init__(self, all_pyside2_modules):
        UIParser.__init__(self, qtproxies.QtCore, qtproxies.QtGui, qtproxies.QtWidgets, CompilerCreatorPolicy(all_pyside2_modules))

    def reset(self):
        qtproxies.i18n_strings = []
        UIParser.reset(self)

    def setContext(self, context):
        qtproxies.i18n_context = context

    def createToplevelWidget(self, classname, widgetname):
        indenter = getIndenter()
        indenter.level = 0
        indenter.write('from PySide2 import QtCore, QtGui, QtWidgets')
        indenter.write('')
        indenter.write('class Ui_%s(object):' % self.uiname)
        indenter.indent()
        indenter.write('def setupUi(self, %s):' % widgetname)
        indenter.indent()
        w = self.factory.createQObject(classname, widgetname, (), is_attribute=False, no_instantiation=True)
        w.baseclass = classname
        w.uiclass = 'Ui_%s' % self.uiname
        return w

    def setDelayedProps(self):
        write_code('')
        write_code('self.retranslateUi(%s)' % self.toplevelWidget)
        UIParser.setDelayedProps(self)

    def finalize(self):
        indenter = getIndenter()
        indenter.level = 1
        indenter.write('')
        indenter.write('def retranslateUi(self, %s):' % self.toplevelWidget)
        indenter.indent()
        if qtproxies.i18n_strings:
            for s in qtproxies.i18n_strings:
                indenter.write(s)
        else:
            indenter.write('pass')
        indenter.dedent()
        indenter.dedent()
        self._resources = self.resources

    def compileUi(self, input_stream, output_stream, from_imports):
        createCodeIndenter(output_stream)
        w = self.parse(input_stream)
        indenter = getIndenter()
        indenter.write('')
        self.factory._cpolicy._writeOutImports()
        for res in self._resources:
            write_import(res, from_imports)
        return {'widgetname': str(w), 'uiclass': w.uiclass, 'baseclass': w.baseclass}