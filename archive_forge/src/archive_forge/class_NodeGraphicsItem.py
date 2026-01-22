import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
class NodeGraphicsItem(GraphicsObject):

    def __init__(self, node):
        GraphicsObject.__init__(self)
        self.pen = fn.mkPen(0, 0, 0)
        self.selectPen = fn.mkPen(200, 200, 200, width=2)
        self.brush = fn.mkBrush(200, 200, 200, 150)
        self.hoverBrush = fn.mkBrush(200, 200, 200, 200)
        self.selectBrush = fn.mkBrush(200, 200, 255, 200)
        self.hovered = False
        self.node = node
        flags = self.GraphicsItemFlag.ItemIsMovable | self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsFocusable | self.GraphicsItemFlag.ItemSendsGeometryChanges
        self.setFlags(flags)
        self.bounds = QtCore.QRectF(0, 0, 100, 100)
        self.nameItem = TextItem(self.node.name(), self, self.labelChanged)
        self.nameItem.setDefaultTextColor(QtGui.QColor(50, 50, 50))
        self.nameItem.moveBy(self.bounds.width() / 2.0 - self.nameItem.boundingRect().width() / 2.0, 0)
        self._titleOffset = 25
        self._nodeOffset = 12
        self.updateTerminals()
        self.menu = None
        self.buildMenu()

    def setTitleOffset(self, new_offset):
        """
        This method sets the rendering offset introduced after the title of the node.
        This method automatically updates the terminal labels. The default for this value is 25px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
        self._titleOffset = new_offset
        self.updateTerminals()

    def titleOffset(self):
        """
        This method returns the current title offset in use.

        :returns: The offset in px.
        """
        return self._titleOffset

    def setTerminalOffset(self, new_offset):
        """
        This method sets the rendering offset introduced after every terminal of the node.
        This method automatically updates the terminal labels. The default for this value is 12px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
        self._nodeOffset = new_offset
        self.updateTerminals()

    def terminalOffset(self):
        """
        This method returns the current terminal offset in use.

        :returns: The offset in px.
        """
        return self._nodeOffset

    def labelChanged(self):
        newName = self.nameItem.toPlainText()
        if newName != self.node.name():
            self.node.rename(newName)
        bounds = self.boundingRect()
        self.nameItem.setPos(bounds.width() / 2.0 - self.nameItem.boundingRect().width() / 2.0, 0)

    def setPen(self, *args, **kwargs):
        self.pen = fn.mkPen(*args, **kwargs)
        self.update()

    def setBrush(self, brush):
        self.brush = brush
        self.update()

    def updateTerminals(self):
        self.terminals = {}
        inp = self.node.inputs()
        out = self.node.outputs()
        maxNode = max(len(inp), len(out))
        newHeight = self._titleOffset + maxNode * self._nodeOffset
        if not self.bounds.height() == newHeight:
            self.bounds.setHeight(newHeight)
            self.update()
        y = self._titleOffset
        for i, t in inp.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            item.setAnchor(0, y)
            self.terminals[i] = (t, item)
            y += self._nodeOffset
        y = self._titleOffset
        for i, t in out.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            item.setZValue(self.zValue())
            item.setAnchor(self.bounds.width(), y)
            self.terminals[i] = (t, item)
            y += self._nodeOffset

    def boundingRect(self):
        return self.bounds.adjusted(-5, -5, 5, 5)

    def paint(self, p, *args):
        p.setPen(self.pen)
        if self.isSelected():
            p.setPen(self.selectPen)
            p.setBrush(self.selectBrush)
        else:
            p.setPen(self.pen)
            if self.hovered:
                p.setBrush(self.hoverBrush)
            else:
                p.setBrush(self.brush)
        p.drawRect(self.bounds)

    def mousePressEvent(self, ev):
        ev.ignore()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            sel = self.isSelected()
            self.setSelected(True)
            if not sel and self.isSelected():
                self.update()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.setPos(self.pos() + self.mapToParent(ev.pos()) - self.mapToParent(ev.lastPos()))

    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
            self.hovered = True
        else:
            self.hovered = False
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Delete or ev.key() == QtCore.Qt.Key.Key_Backspace:
            ev.accept()
            if not self.node._allowRemove:
                return
            self.node.close()
        else:
            ev.ignore()

    def itemChange(self, change, val):
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            for k, t in self.terminals.items():
                t[1].nodeMoved()
        return GraphicsObject.itemChange(self, change, val)

    def getMenu(self):
        return self.menu

    def raiseContextMenu(self, ev):
        menu = self.scene().addParentContextMenus(self, self.getMenu(), ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def buildMenu(self):
        self.menu = QtWidgets.QMenu()
        self.menu.setTitle(translate('Context Menu', 'Node'))
        a = self.menu.addAction(translate('Context Menu', 'Add input'), self.addInputFromMenu)
        if not self.node._allowAddInput:
            a.setEnabled(False)
        a = self.menu.addAction(translate('Context Menu', 'Add output'), self.addOutputFromMenu)
        if not self.node._allowAddOutput:
            a.setEnabled(False)
        a = self.menu.addAction(translate('Context Menu', 'Remove node'), self.node.close)
        if not self.node._allowRemove:
            a.setEnabled(False)

    def addInputFromMenu(self):
        self.node.addInput(renamable=True, removable=True, multiable=True)

    def addOutputFromMenu(self):
        self.node.addOutput(renamable=True, removable=True, multiable=False)