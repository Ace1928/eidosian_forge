from PySide2 import QtCore, QtGui, QtWidgets
@staticmethod
def updateComboBox(comboBox):
    if comboBox.findText(comboBox.currentText()) == -1:
        comboBox.addItem(comboBox.currentText())