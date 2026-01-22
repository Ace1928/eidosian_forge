from PySide2.QtCore import (Signal, QDataStream, QMutex, QMutexLocker,
from PySide2.QtGui import QIntValidator
from PySide2.QtWidgets import (QApplication, QDialogButtonBox, QGridLayout,
from PySide2.QtNetwork import (QAbstractSocket, QHostAddress, QNetworkInterface,
class BlockingClient(QWidget):

    def __init__(self, parent=None):
        super(BlockingClient, self).__init__(parent)
        self.thread = FortuneThread()
        self.currentFortune = ''
        hostLabel = QLabel('&Server name:')
        portLabel = QLabel('S&erver port:')
        for ipAddress in QNetworkInterface.allAddresses():
            if ipAddress != QHostAddress.LocalHost and ipAddress.toIPv4Address() != 0:
                break
        else:
            ipAddress = QHostAddress(QHostAddress.LocalHost)
        ipAddress = ipAddress.toString()
        self.hostLineEdit = QLineEdit(ipAddress)
        self.portLineEdit = QLineEdit()
        self.portLineEdit.setValidator(QIntValidator(1, 65535, self))
        hostLabel.setBuddy(self.hostLineEdit)
        portLabel.setBuddy(self.portLineEdit)
        self.statusLabel = QLabel('This example requires that you run the Fortune Server example as well.')
        self.statusLabel.setWordWrap(True)
        self.getFortuneButton = QPushButton('Get Fortune')
        self.getFortuneButton.setDefault(True)
        self.getFortuneButton.setEnabled(False)
        quitButton = QPushButton('Quit')
        buttonBox = QDialogButtonBox()
        buttonBox.addButton(self.getFortuneButton, QDialogButtonBox.ActionRole)
        buttonBox.addButton(quitButton, QDialogButtonBox.RejectRole)
        self.getFortuneButton.clicked.connect(self.requestNewFortune)
        quitButton.clicked.connect(self.close)
        self.hostLineEdit.textChanged.connect(self.enableGetFortuneButton)
        self.portLineEdit.textChanged.connect(self.enableGetFortuneButton)
        self.thread.newFortune.connect(self.showFortune)
        self.thread.error.connect(self.displayError)
        mainLayout = QGridLayout()
        mainLayout.addWidget(hostLabel, 0, 0)
        mainLayout.addWidget(self.hostLineEdit, 0, 1)
        mainLayout.addWidget(portLabel, 1, 0)
        mainLayout.addWidget(self.portLineEdit, 1, 1)
        mainLayout.addWidget(self.statusLabel, 2, 0, 1, 2)
        mainLayout.addWidget(buttonBox, 3, 0, 1, 2)
        self.setLayout(mainLayout)
        self.setWindowTitle('Blocking Fortune Client')
        self.portLineEdit.setFocus()

    def requestNewFortune(self):
        self.getFortuneButton.setEnabled(False)
        self.thread.requestNewFortune(self.hostLineEdit.text(), int(self.portLineEdit.text()))

    def showFortune(self, nextFortune):
        if nextFortune == self.currentFortune:
            self.requestNewFortune()
            return
        self.currentFortune = nextFortune
        self.statusLabel.setText(self.currentFortune)
        self.getFortuneButton.setEnabled(True)

    def displayError(self, socketError, message):
        if socketError == QAbstractSocket.HostNotFoundError:
            QMessageBox.information(self, 'Blocking Fortune Client', 'The host was not found. Please check the host and port settings.')
        elif socketError == QAbstractSocket.ConnectionRefusedError:
            QMessageBox.information(self, 'Blocking Fortune Client', 'The connection was refused by the peer. Make sure the fortune server is running, and check that the host name and port settings are correct.')
        else:
            QMessageBox.information(self, 'Blocking Fortune Client', 'The following error occurred: %s.' % message)
        self.getFortuneButton.setEnabled(True)

    def enableGetFortuneButton(self):
        self.getFortuneButton.setEnabled(self.hostLineEdit.text() != '' and self.portLineEdit.text() != '')