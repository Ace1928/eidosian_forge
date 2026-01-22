import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QToolBox(QFrame):

    def addItem(self, *args):
        text = args[-1]
        if isinstance(text, i18n_string):
            i18n_print('%s.setItemText(%s.indexOf(%s), %s)' % (self._uic_name, self._uic_name, args[0], text))
            args = args[:-1] + ('',)
        ProxyClassMember(self, 'addItem', 0)(*args)

    def indexOf(self, page):
        return Literal('%s.indexOf(%s)' % (self, page))

    def layout(self):
        return QtWidgets.QLayout('%s.layout()' % self, False, (), noInstantiation=True)