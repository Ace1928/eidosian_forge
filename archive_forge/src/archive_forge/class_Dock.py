import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
class Dock(QtWidgets.QWidget):
    sigStretchChanged = QtCore.Signal()
    sigClosed = QtCore.Signal(object)

    def __init__(self, name, area=None, size=(10, 10), widget=None, hideTitle=False, autoOrientation=True, label=None, **kargs):
        QtWidgets.QWidget.__init__(self)
        self.dockdrop = DockDrop(self)
        self._container = None
        self._name = name
        self.area = area
        self.label = label
        if self.label is None:
            self.label = DockLabel(name, **kargs)
        self.label.dock = self
        if self.label.isClosable():
            self.label.sigCloseClicked.connect(self.close)
        self.labelHidden = False
        self.moveLabel = True
        self.autoOrient = autoOrientation
        self.orientation = 'horizontal'
        self.topLayout = QtWidgets.QGridLayout()
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.setSpacing(0)
        self.setLayout(self.topLayout)
        self.topLayout.addWidget(self.label, 0, 1)
        self.widgetArea = QtWidgets.QWidget()
        self.topLayout.addWidget(self.widgetArea, 1, 1)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.widgetArea.setLayout(self.layout)
        self.widgetArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.widgets = []
        self.currentRow = 0
        self.dockdrop.raiseOverlay()
        self.hStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n            border-top-left-radius: 0px;\n            border-top-right-radius: 0px;\n            border-top-width: 0px;\n        }'
        self.vStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n            border-top-left-radius: 0px;\n            border-bottom-left-radius: 0px;\n            border-left-width: 0px;\n        }'
        self.nStyle = '\n        Dock > QWidget {\n            border: 1px solid #000;\n            border-radius: 5px;\n        }'
        self.dragStyle = '\n        Dock > QWidget {\n            border: 4px solid #00F;\n            border-radius: 5px;\n        }'
        self.setAutoFillBackground(False)
        self.widgetArea.setStyleSheet(self.hStyle)
        self.setStretch(*size)
        if widget is not None:
            self.addWidget(widget)
        if hideTitle:
            self.hideTitleBar()

    def implements(self, name=None):
        if name is None:
            return ['dock']
        else:
            return name == 'dock'

    def setStretch(self, x=None, y=None):
        """
        Set the 'target' size for this Dock.
        The actual size will be determined by comparing this Dock's
        stretch value to the rest of the docks it shares space with.
        """
        if x is None:
            x = 0
        if y is None:
            y = 0
        self._stretch = (x, y)
        self.sigStretchChanged.emit()

    def stretch(self):
        return self._stretch

    def hideTitleBar(self):
        """
        Hide the title bar for this Dock.
        This will prevent the Dock being moved by the user.
        """
        self.label.hide()
        self.labelHidden = True
        self.dockdrop.removeAllowedArea('center')
        self.updateStyle()

    def showTitleBar(self):
        """
        Show the title bar for this Dock.
        """
        self.label.show()
        self.labelHidden = False
        self.dockdrop.addAllowedArea('center')
        self.updateStyle()

    def title(self):
        """
        Gets the text displayed in the title bar for this dock.
        """
        return self.label.text()

    def setTitle(self, text):
        """
        Sets the text displayed in title bar for this Dock.
        """
        self.label.setText(text)

    def setOrientation(self, o='auto', force=False):
        """
        Sets the orientation of the title bar for this Dock.
        Must be one of 'auto', 'horizontal', or 'vertical'.
        By default ('auto'), the orientation is determined
        based on the aspect ratio of the Dock.
        """
        if self.container() is None:
            return
        if o == 'auto' and self.autoOrient:
            if self.container().type() == 'tab':
                o = 'horizontal'
            elif self.width() > self.height() * 1.5:
                o = 'vertical'
            else:
                o = 'horizontal'
        if force or self.orientation != o:
            self.orientation = o
            self.label.setOrientation(o)
            self.updateStyle()

    def updateStyle(self):
        if self.labelHidden:
            self.widgetArea.setStyleSheet(self.nStyle)
        elif self.orientation == 'vertical':
            self.label.setOrientation('vertical')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 1, 0)
            self.widgetArea.setStyleSheet(self.vStyle)
        else:
            self.label.setOrientation('horizontal')
            if self.moveLabel:
                self.topLayout.addWidget(self.label, 0, 1)
            self.widgetArea.setStyleSheet(self.hStyle)

    def resizeEvent(self, ev):
        self.setOrientation()
        self.dockdrop.resizeOverlay(self.size())

    def name(self):
        return self._name

    def addWidget(self, widget, row=None, col=0, rowspan=1, colspan=1):
        """
        Add a new widget to the interior of this Dock.
        Each Dock uses a QGridLayout to arrange widgets within.
        """
        if row is None:
            row = self.currentRow
        self.currentRow = max(row + 1, self.currentRow)
        self.widgets.append(widget)
        self.layout.addWidget(widget, row, col, rowspan, colspan)
        self.dockdrop.raiseOverlay()

    def startDrag(self):
        self.drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        self.drag.setMimeData(mime)
        self.widgetArea.setStyleSheet(self.dragStyle)
        self.update()
        action = self.drag.exec() if hasattr(self.drag, 'exec') else self.drag.exec_()
        self.updateStyle()

    def float(self):
        self.area.floatDock(self)

    def container(self):
        return self._container

    def containerChanged(self, c):
        if self._container is not None:
            self._container.apoptose(propagate=False)
        self._container = c
        if c is None:
            self.area = None
        else:
            self.area = c.area
            if c.type() != 'tab':
                self.moveLabel = True
                self.label.setDim(False)
            else:
                self.moveLabel = False
            self.setOrientation(force=True)

    def raiseDock(self):
        """If this Dock is stacked underneath others, raise it to the top."""
        self.container().raiseDock(self)

    def close(self):
        """Remove this dock from the DockArea it lives inside."""
        if self._container is None:
            warnings.warn(f'Cannot close dock {self} because it is not open.', RuntimeWarning, stacklevel=2)
            return
        self.setParent(None)
        QtWidgets.QLabel.close(self.label)
        self.label.setParent(None)
        self._container.apoptose()
        self._container = None
        self.sigClosed.emit(self)

    def __repr__(self):
        return '<Dock %s %s>' % (self.name(), self.stretch())

    def dragEnterEvent(self, *args):
        self.dockdrop.dragEnterEvent(*args)

    def dragMoveEvent(self, *args):
        self.dockdrop.dragMoveEvent(*args)

    def dragLeaveEvent(self, *args):
        self.dockdrop.dragLeaveEvent(*args)

    def dropEvent(self, *args):
        self.dockdrop.dropEvent(*args)