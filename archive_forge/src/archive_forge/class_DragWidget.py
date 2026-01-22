from PySide2.QtCore import QFile, QIODevice, QMimeData, QPoint, Qt, QTextStream
from PySide2.QtGui import QDrag, QPalette, QPixmap
from PySide2.QtWidgets import QApplication, QFrame, QLabel, QWidget
import draggabletext_rc
class DragWidget(QWidget):

    def __init__(self, parent=None):
        super(DragWidget, self).__init__(parent)
        dictionaryFile = QFile(':/dictionary/words.txt')
        dictionaryFile.open(QIODevice.ReadOnly)
        x = 5
        y = 5
        for word in QTextStream(dictionaryFile).readAll().split():
            wordLabel = DragLabel(word, self)
            wordLabel.move(x, y)
            wordLabel.show()
            x += wordLabel.width() + 2
            if x >= 195:
                x = 5
                y += wordLabel.height() + 2
        newPalette = self.palette()
        newPalette.setColor(QPalette.Window, Qt.white)
        self.setPalette(newPalette)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, max(200, y))
        self.setWindowTitle('Draggable Text')

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            if event.source() in self.children():
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            mime = event.mimeData()
            pieces = mime.text().split()
            position = event.pos()
            hotSpot = QPoint()
            hotSpotPos = mime.data('application/x-hotspot').split(' ')
            if len(hotSpotPos) == 2:
                hotSpot.setX(hotSpotPos[0].toInt()[0])
                hotSpot.setY(hotSpotPos[1].toInt()[0])
            for piece in pieces:
                newLabel = DragLabel(piece, self)
                newLabel.move(position - hotSpot)
                newLabel.show()
                position += QPoint(newLabel.width(), 0)
            if event.source() in self.children():
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()