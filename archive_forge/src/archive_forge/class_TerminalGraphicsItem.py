import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
class TerminalGraphicsItem(GraphicsObject):

    def __init__(self, term, parent=None):
        self.term = term
        GraphicsObject.__init__(self, parent)
        self.brush = fn.mkBrush(0, 0, 0)
        self.box = QtWidgets.QGraphicsRectItem(0, 0, 10, 10, self)
        on_update = self.labelChanged if self.term.isRenamable() else None
        self.label = TextItem(self.term.name(), self, on_update)
        self.label.setScale(0.7)
        self.newConnection = None
        self.setFiltersChildEvents(True)
        if self.term.isRenamable():
            self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        self.setZValue(1)
        self.menu = None

    def labelChanged(self):
        newName = self.label.toPlainText()
        if newName != self.term.name():
            self.term.rename(newName)

    def termRenamed(self, name):
        self.label.setPlainText(name)

    def setBrush(self, brush):
        self.brush = brush
        self.box.setBrush(brush)

    def disconnect(self, target):
        self.term.disconnectFrom(target.term)

    def boundingRect(self):
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        return br | lr

    def paint(self, p, *args):
        pass

    def setAnchor(self, x, y):
        pos = QtCore.QPointF(x, y)
        self.anchorPos = pos
        br = self.box.mapRectToParent(self.box.boundingRect())
        lr = self.label.mapRectToParent(self.label.boundingRect())
        if self.term.isInput():
            self.box.setPos(pos.x(), pos.y() - br.height() / 2.0)
            self.label.setPos(pos.x() + br.width(), pos.y() - lr.height() / 2.0)
        else:
            self.box.setPos(pos.x() - br.width(), pos.y() - br.height() / 2.0)
            self.label.setPos(pos.x() - br.width() - lr.width(), pos.y() - lr.height() / 2.0)
        self.updateConnections()

    def updateConnections(self):
        for t, c in self.term.connections().items():
            c.updateLine()

    def mousePressEvent(self, ev):
        ev.ignore()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.label.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate('Context Menu', 'Terminal'))
            remAct = QtGui.QAction(translate('Context Menu', 'Remove terminal'), self.menu)
            remAct.triggered.connect(self.removeSelf)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
            if not self.term.isRemovable():
                remAct.setEnabled(False)
            multiAct = QtGui.QAction(translate('Context Menu', 'Multi-value'), self.menu)
            multiAct.setCheckable(True)
            multiAct.setChecked(self.term.isMultiValue())
            multiAct.setEnabled(self.term.isMultiable())
            multiAct.triggered.connect(self.toggleMulti)
            self.menu.addAction(multiAct)
            self.menu.multiAct = multiAct
            if self.term.isMultiable():
                multiAct.setEnabled = False
        return self.menu

    def toggleMulti(self):
        multi = self.menu.multiAct.isChecked()
        self.term.setMultiValue(multi)

    def removeSelf(self):
        self.term.node().removeTerminal(self.term)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            if self.newConnection is None:
                self.newConnection = ConnectionItem(self)
                self.getViewBox().addItem(self.newConnection)
            self.newConnection.setTarget(self.mapToView(ev.pos()))
        elif ev.isFinish():
            if self.newConnection is not None:
                items = self.scene().items(ev.scenePos())
                gotTarget = False
                for i in items:
                    if isinstance(i, TerminalGraphicsItem):
                        self.newConnection.setTarget(i)
                        try:
                            self.term.connectTo(i.term, self.newConnection)
                            gotTarget = True
                        except:
                            self.scene().removeItem(self.newConnection)
                            self.newConnection = None
                            raise
                        break
                if not gotTarget:
                    self.newConnection.close()
                self.newConnection = None
        elif self.newConnection is not None:
            self.newConnection.setTarget(self.mapToView(ev.pos()))

    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            self.box.setBrush(fn.mkBrush('w'))
        else:
            self.box.setBrush(self.brush)
        self.update()

    def connectPoint(self):
        return self.mapToView(self.mapFromItem(self.box, self.box.boundingRect().center()))

    def nodeMoved(self):
        for t, item in self.term.connections().items():
            item.updateLine()