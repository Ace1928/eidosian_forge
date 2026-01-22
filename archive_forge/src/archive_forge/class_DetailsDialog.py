from PySide2 import QtCore, QtGui, QtWidgets, QtPrintSupport
class DetailsDialog(QtWidgets.QDialog):

    def __init__(self, title, parent):
        super(DetailsDialog, self).__init__(parent)
        self.items = ('T-shirt', 'Badge', 'Reference book', 'Coffee cup')
        nameLabel = QtWidgets.QLabel('Name:')
        addressLabel = QtWidgets.QLabel('Address:')
        addressLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.nameEdit = QtWidgets.QLineEdit()
        self.addressEdit = QtWidgets.QTextEdit()
        self.offersCheckBox = QtWidgets.QCheckBox('Send information about products and special offers:')
        self.setupItemsTable()
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.verify)
        buttonBox.rejected.connect(self.reject)
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(nameLabel, 0, 0)
        mainLayout.addWidget(self.nameEdit, 0, 1)
        mainLayout.addWidget(addressLabel, 1, 0)
        mainLayout.addWidget(self.addressEdit, 1, 1)
        mainLayout.addWidget(self.itemsTable, 0, 2, 2, 1)
        mainLayout.addWidget(self.offersCheckBox, 2, 1, 1, 2)
        mainLayout.addWidget(buttonBox, 3, 0, 1, 3)
        self.setLayout(mainLayout)
        self.setWindowTitle(title)

    def setupItemsTable(self):
        self.itemsTable = QtWidgets.QTableWidget(len(self.items), 2)
        for row, item in enumerate(self.items):
            name = QtWidgets.QTableWidgetItem(item)
            name.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.itemsTable.setItem(row, 0, name)
            quantity = QtWidgets.QTableWidgetItem('1')
            self.itemsTable.setItem(row, 1, quantity)

    def orderItems(self):
        orderList = []
        for row in range(len(self.items)):
            text = self.itemsTable.item(row, 0).text()
            quantity = int(self.itemsTable.item(row, 1).data(QtCore.Qt.DisplayRole))
            orderList.append((text, max(0, quantity)))
        return orderList

    def senderName(self):
        return self.nameEdit.text()

    def senderAddress(self):
        return self.addressEdit.toPlainText()

    def sendOffers(self):
        return self.offersCheckBox.isChecked()

    def verify(self):
        if self.nameEdit.text() and self.addressEdit.toPlainText():
            self.accept()
            return
        answer = QtWidgets.QMessageBox.warning(self, 'Incomplete Form', 'The form does not contain all the necessary information.\nDo you want to discard it?', QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            self.reject()