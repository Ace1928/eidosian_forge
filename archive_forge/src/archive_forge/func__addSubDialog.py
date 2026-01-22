from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def _addSubDialog(self, dlg):
    self._prepareNesting()
    bar, btn = dlg._extractWidgets()
    inserted = False
    for i, bar2 in enumerate(self._subBars):
        if bar2.hidden:
            self._subBars.pop(i)
            bar2.hide()
            bar2.setParent(None)
            self._subBars.insert(i, bar)
            inserted = True
            break
    if not inserted:
        self._subBars.append(bar)
    while self.nestedLayout.count() > 0:
        self.nestedLayout.takeAt(0)
    for b in self._subBars:
        self.nestedLayout.addWidget(b)