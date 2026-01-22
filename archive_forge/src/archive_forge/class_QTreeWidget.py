import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QTreeWidget(QTreeView):
    isSortingEnabled = i18n_func('isSortingEnabled')
    setSortingEnabled = i18n_void_func('setSortingEnabled')

    def headerItem(self):
        return QtWidgets.QWidget('%s.headerItem()' % self, False, (), noInstantiation=True)

    def topLevelItem(self, index):
        return QtWidgets.QTreeWidgetItem('%s.topLevelItem(%i)' % (self, index), False, (), noInstantiation=True)