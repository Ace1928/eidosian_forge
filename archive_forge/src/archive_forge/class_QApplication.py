import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QApplication(QtCore.QObject):

    def translate(uiname, text, disambig, encoding):
        return i18n_string(text or '', disambig)
    translate = staticmethod(translate)