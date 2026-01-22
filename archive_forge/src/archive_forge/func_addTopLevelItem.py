from ..Qt import QtCore, QtWidgets
def addTopLevelItem(self, item):
    QtWidgets.QTreeWidget.addTopLevelItem(self, item)
    self.informTreeWidgetChange(item)