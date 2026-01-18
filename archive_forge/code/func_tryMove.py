import random
from PySide2 import QtCore, QtGui, QtWidgets
def tryMove(self, newPiece, newX, newY):
    for i in range(4):
        x = newX + newPiece.x(i)
        y = newY - newPiece.y(i)
        if x < 0 or x >= TetrixBoard.BoardWidth or y < 0 or (y >= TetrixBoard.BoardHeight):
            return False
        if self.shapeAt(x, y) != NoShape:
            return False
    self.curPiece = newPiece
    self.curX = newX
    self.curY = newY
    self.update()
    return True