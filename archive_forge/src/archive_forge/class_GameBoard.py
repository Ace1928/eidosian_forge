import sys
import math
import random
from PySide2 import QtCore, QtGui, QtWidgets
class GameBoard(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        quit = QtWidgets.QPushButton('&Quit')
        quit.setFont(QtGui.QFont('Times', 18, QtGui.QFont.Bold))
        self.connect(quit, QtCore.SIGNAL('clicked()'), QtWidgets.qApp, QtCore.SLOT('quit()'))
        angle = LCDRange('ANGLE')
        angle.setRange(5, 70)
        force = LCDRange('FORCE')
        force.setRange(10, 50)
        self.cannonField = CannonField()
        self.connect(angle, QtCore.SIGNAL('valueChanged(int)'), self.cannonField.setAngle)
        self.connect(self.cannonField, QtCore.SIGNAL('angleChanged(int)'), angle.setValue)
        self.connect(force, QtCore.SIGNAL('valueChanged(int)'), self.cannonField.setForce)
        self.connect(self.cannonField, QtCore.SIGNAL('forceChanged(int)'), force.setValue)
        self.connect(self.cannonField, QtCore.SIGNAL('hit()'), self.hit)
        self.connect(self.cannonField, QtCore.SIGNAL('missed()'), self.missed)
        shoot = QtWidgets.QPushButton('&Shoot')
        shoot.setFont(QtGui.QFont('Times', 18, QtGui.QFont.Bold))
        self.connect(shoot, QtCore.SIGNAL('clicked()'), self.fire)
        self.connect(self.cannonField, QtCore.SIGNAL('canShoot(bool)'), shoot, QtCore.SLOT('setEnabled(bool)'))
        restart = QtWidgets.QPushButton('&New Game')
        restart.setFont(QtGui.QFont('Times', 18, QtGui.QFont.Bold))
        self.connect(restart, QtCore.SIGNAL('clicked()'), self.newGame)
        self.hits = QtWidgets.QLCDNumber(2)
        self.shotsLeft = QtWidgets.QLCDNumber(2)
        hitsLabel = QtWidgets.QLabel('HITS')
        shotsLeftLabel = QtWidgets.QLabel('SHOTS LEFT')
        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(shoot)
        topLayout.addWidget(self.hits)
        topLayout.addWidget(hitsLabel)
        topLayout.addWidget(self.shotsLeft)
        topLayout.addWidget(shotsLeftLabel)
        topLayout.addStretch(1)
        topLayout.addWidget(restart)
        leftLayout = QtWidgets.QVBoxLayout()
        leftLayout.addWidget(angle)
        leftLayout.addWidget(force)
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(quit, 0, 0)
        gridLayout.addLayout(topLayout, 0, 1)
        gridLayout.addLayout(leftLayout, 1, 0)
        gridLayout.addWidget(self.cannonField, 1, 1, 2, 1)
        gridLayout.setColumnStretch(1, 10)
        self.setLayout(gridLayout)
        angle.setValue(60)
        force.setValue(25)
        angle.setFocus()
        self.newGame()

    @QtCore.Slot()
    def fire(self):
        if self.cannonField.gameOver() or self.cannonField.isShooting():
            return
        self.shotsLeft.display(self.shotsLeft.intValue() - 1)
        self.cannonField.shoot()

    @QtCore.Slot()
    def hit(self):
        self.hits.display(self.hits.intValue() + 1)
        if self.shotsLeft.intValue() == 0:
            self.cannonField.setGameOver()
        else:
            self.cannonField.newTarget()

    @QtCore.Slot()
    def missed(self):
        if self.shotsLeft.intValue() == 0:
            self.cannonField.setGameOver()

    @QtCore.Slot()
    def newGame(self):
        self.shotsLeft.display(15)
        self.hits.display(0)
        self.cannonField.restartGame()
        self.cannonField.newTarget()