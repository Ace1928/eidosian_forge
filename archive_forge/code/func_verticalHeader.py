import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def verticalHeader(self):
    return QtWidgets.QHeaderView('%s.verticalHeader()' % self, False, (), noInstantiation=True)