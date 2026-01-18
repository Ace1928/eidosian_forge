import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def topLevelItem(self, index):
    return QtWidgets.QTreeWidgetItem('%s.topLevelItem(%i)' % (self, index), False, (), noInstantiation=True)