from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QDialog, QLabel, QTextEdit, QLineEdit,
class AddDialogWidget(QDialog):
    """ A dialog to add a new address to the addressbook. """

    def __init__(self, parent=None):
        super(AddDialogWidget, self).__init__(parent)
        nameLabel = QLabel('Name')
        addressLabel = QLabel('Address')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.nameText = QLineEdit()
        self.addressText = QTextEdit()
        grid = QGridLayout()
        grid.setColumnStretch(1, 2)
        grid.addWidget(nameLabel, 0, 0)
        grid.addWidget(self.nameText, 0, 1)
        grid.addWidget(addressLabel, 1, 0, Qt.AlignLeft | Qt.AlignTop)
        grid.addWidget(self.addressText, 1, 1, Qt.AlignLeft)
        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(buttonBox)
        self.setLayout(layout)
        self.setWindowTitle('Add a Contact')
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    @property
    def name(self):
        return self.nameText.text()

    @property
    def address(self):
        return self.addressText.toPlainText()