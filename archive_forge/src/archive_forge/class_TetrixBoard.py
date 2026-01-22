import random
from PySide2 import QtCore, QtGui, QtWidgets
class TetrixBoard(QtWidgets.QFrame):
    BoardWidth = 10
    BoardHeight = 22
    scoreChanged = QtCore.Signal(int)
    levelChanged = QtCore.Signal(int)
    linesRemovedChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(TetrixBoard, self).__init__(parent)
        self.timer = QtCore.QBasicTimer()
        self.nextPieceLabel = None
        self.isWaitingAfterLine = False
        self.curPiece = TetrixPiece()
        self.nextPiece = TetrixPiece()
        self.curX = 0
        self.curY = 0
        self.numLinesRemoved = 0
        self.numPiecesDropped = 0
        self.score = 0
        self.level = 0
        self.board = None
        self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.isStarted = False
        self.isPaused = False
        self.clearBoard()
        self.nextPiece.setRandomShape()

    def shapeAt(self, x, y):
        return self.board[y * TetrixBoard.BoardWidth + x]

    def setShapeAt(self, x, y, shape):
        self.board[y * TetrixBoard.BoardWidth + x] = shape

    def timeoutTime(self):
        return 1000 / (1 + self.level)

    def squareWidth(self):
        return self.contentsRect().width() / TetrixBoard.BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() / TetrixBoard.BoardHeight

    def setNextPieceLabel(self, label):
        self.nextPieceLabel = label

    def sizeHint(self):
        return QtCore.QSize(TetrixBoard.BoardWidth * 15 + self.frameWidth() * 2, TetrixBoard.BoardHeight * 15 + self.frameWidth() * 2)

    def minimumSizeHint(self):
        return QtCore.QSize(TetrixBoard.BoardWidth * 5 + self.frameWidth() * 2, TetrixBoard.BoardHeight * 5 + self.frameWidth() * 2)

    def start(self):
        if self.isPaused:
            return
        self.isStarted = True
        self.isWaitingAfterLine = False
        self.numLinesRemoved = 0
        self.numPiecesDropped = 0
        self.score = 0
        self.level = 1
        self.clearBoard()
        self.linesRemovedChanged.emit(self.numLinesRemoved)
        self.scoreChanged.emit(self.score)
        self.levelChanged.emit(self.level)
        self.newPiece()
        self.timer.start(self.timeoutTime(), self)

    def pause(self):
        if not self.isStarted:
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
        else:
            self.timer.start(self.timeoutTime(), self)
        self.update()

    def paintEvent(self, event):
        super(TetrixBoard, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        rect = self.contentsRect()
        if self.isPaused:
            painter.drawText(rect, QtCore.Qt.AlignCenter, 'Pause')
            return
        boardTop = rect.bottom() - TetrixBoard.BoardHeight * self.squareHeight()
        for i in range(TetrixBoard.BoardHeight):
            for j in range(TetrixBoard.BoardWidth):
                shape = self.shapeAt(j, TetrixBoard.BoardHeight - i - 1)
                if shape != NoShape:
                    self.drawSquare(painter, rect.left() + j * self.squareWidth(), boardTop + i * self.squareHeight(), shape)
        if self.curPiece.shape() != NoShape:
            for i in range(4):
                x = self.curX + self.curPiece.x(i)
                y = self.curY - self.curPiece.y(i)
                self.drawSquare(painter, rect.left() + x * self.squareWidth(), boardTop + (TetrixBoard.BoardHeight - y - 1) * self.squareHeight(), self.curPiece.shape())

    def keyPressEvent(self, event):
        if not self.isStarted or self.isPaused or self.curPiece.shape() == NoShape:
            super(TetrixBoard, self).keyPressEvent(event)
            return
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self.tryMove(self.curPiece, self.curX - 1, self.curY)
        elif key == QtCore.Qt.Key_Right:
            self.tryMove(self.curPiece, self.curX + 1, self.curY)
        elif key == QtCore.Qt.Key_Down:
            self.tryMove(self.curPiece.rotatedRight(), self.curX, self.curY)
        elif key == QtCore.Qt.Key_Up:
            self.tryMove(self.curPiece.rotatedLeft(), self.curX, self.curY)
        elif key == QtCore.Qt.Key_Space:
            self.dropDown()
        elif key == QtCore.Qt.Key_D:
            self.oneLineDown()
        else:
            super(TetrixBoard, self).keyPressEvent(event)

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self.isWaitingAfterLine:
                self.isWaitingAfterLine = False
                self.newPiece()
                self.timer.start(self.timeoutTime(), self)
            else:
                self.oneLineDown()
        else:
            super(TetrixBoard, self).timerEvent(event)

    def clearBoard(self):
        self.board = [NoShape for i in range(TetrixBoard.BoardHeight * TetrixBoard.BoardWidth)]

    def dropDown(self):
        dropHeight = 0
        newY = self.curY
        while newY > 0:
            if not self.tryMove(self.curPiece, self.curX, newY - 1):
                break
            newY -= 1
            dropHeight += 1
        self.pieceDropped(dropHeight)

    def oneLineDown(self):
        if not self.tryMove(self.curPiece, self.curX, self.curY - 1):
            self.pieceDropped(0)

    def pieceDropped(self, dropHeight):
        for i in range(4):
            x = self.curX + self.curPiece.x(i)
            y = self.curY - self.curPiece.y(i)
            self.setShapeAt(x, y, self.curPiece.shape())
        self.numPiecesDropped += 1
        if self.numPiecesDropped % 25 == 0:
            self.level += 1
            self.timer.start(self.timeoutTime(), self)
            self.levelChanged.emit(self.level)
        self.score += dropHeight + 7
        self.scoreChanged.emit(self.score)
        self.removeFullLines()
        if not self.isWaitingAfterLine:
            self.newPiece()

    def removeFullLines(self):
        numFullLines = 0
        for i in range(TetrixBoard.BoardHeight - 1, -1, -1):
            lineIsFull = True
            for j in range(TetrixBoard.BoardWidth):
                if self.shapeAt(j, i) == NoShape:
                    lineIsFull = False
                    break
            if lineIsFull:
                numFullLines += 1
                for k in range(TetrixBoard.BoardHeight - 1):
                    for j in range(TetrixBoard.BoardWidth):
                        self.setShapeAt(j, k, self.shapeAt(j, k + 1))
                for j in range(TetrixBoard.BoardWidth):
                    self.setShapeAt(j, TetrixBoard.BoardHeight - 1, NoShape)
        if numFullLines > 0:
            self.numLinesRemoved += numFullLines
            self.score += 10 * numFullLines
            self.linesRemovedChanged.emit(self.numLinesRemoved)
            self.scoreChanged.emit(self.score)
            self.timer.start(500, self)
            self.isWaitingAfterLine = True
            self.curPiece.setShape(NoShape)
            self.update()

    def newPiece(self):
        self.curPiece = self.nextPiece
        self.nextPiece.setRandomShape()
        self.showNextPiece()
        self.curX = TetrixBoard.BoardWidth // 2 + 1
        self.curY = TetrixBoard.BoardHeight - 1 + self.curPiece.minY()
        if not self.tryMove(self.curPiece, self.curX, self.curY):
            self.curPiece.setShape(NoShape)
            self.timer.stop()
            self.isStarted = False

    def showNextPiece(self):
        if self.nextPieceLabel is not None:
            return
        dx = self.nextPiece.maxX() - self.nextPiece.minX() + 1
        dy = self.nextPiece.maxY() - self.nextPiece.minY() + 1
        pixmap = QtGui.QPixmap(dx * self.squareWidth(), dy * self.squareHeight())
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(pixmap.rect(), self.nextPieceLabel.palette().background())
        for int in range(4):
            x = self.nextPiece.x(i) - self.nextPiece.minX()
            y = self.nextPiece.y(i) - self.nextPiece.minY()
            self.drawSquare(painter, x * self.squareWidth(), y * self.squareHeight(), self.nextPiece.shape())
        self.nextPieceLabel.setPixmap(pixmap)

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

    def drawSquare(self, painter, x, y, shape):
        colorTable = [0, 13395558, 6736998, 6710988, 13421670, 13395660, 6737100, 14330368]
        color = QtGui.QColor(colorTable[shape])
        painter.fillRect(x + 1, y + 1, self.squareWidth() - 2, self.squareHeight() - 2, color)
        painter.setPen(color.lighter())
        painter.drawLine(x, y + self.squareHeight() - 1, x, y)
        painter.drawLine(x, y, x + self.squareWidth() - 1, y)
        painter.setPen(color.darker())
        painter.drawLine(x + 1, y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + self.squareHeight() - 1)
        painter.drawLine(x + self.squareWidth() - 1, y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + 1)