from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
class OutputFilesPage(QtWidgets.QWizardPage):

    def __init__(self, parent=None):
        super(OutputFilesPage, self).__init__(parent)
        self.setTitle('Output Files')
        self.setSubTitle('Specify where you want the wizard to put the generated skeleton code.')
        self.setPixmap(QtWidgets.QWizard.LogoPixmap, QtGui.QPixmap(':/images/logo3.png'))
        outputDirLabel = QtWidgets.QLabel('&Output directory:')
        self.outputDirLineEdit = QtWidgets.QLineEdit()
        outputDirLabel.setBuddy(self.outputDirLineEdit)
        headerLabel = QtWidgets.QLabel('&Header file name:')
        self.headerLineEdit = QtWidgets.QLineEdit()
        headerLabel.setBuddy(self.headerLineEdit)
        implementationLabel = QtWidgets.QLabel('&Implementation file name:')
        self.implementationLineEdit = QtWidgets.QLineEdit()
        implementationLabel.setBuddy(self.implementationLineEdit)
        self.registerField('outputDir*', self.outputDirLineEdit)
        self.registerField('header*', self.headerLineEdit)
        self.registerField('implementation*', self.implementationLineEdit)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(outputDirLabel, 0, 0)
        layout.addWidget(self.outputDirLineEdit, 0, 1)
        layout.addWidget(headerLabel, 1, 0)
        layout.addWidget(self.headerLineEdit, 1, 1)
        layout.addWidget(implementationLabel, 2, 0)
        layout.addWidget(self.implementationLineEdit, 2, 1)
        self.setLayout(layout)

    def initializePage(self):
        className = self.field('className')
        self.headerLineEdit.setText(className.lower() + '.h')
        self.implementationLineEdit.setText(className.lower() + '.cpp')
        self.outputDirLineEdit.setText(QtCore.QDir.toNativeSeparators(QtCore.QDir.tempPath()))