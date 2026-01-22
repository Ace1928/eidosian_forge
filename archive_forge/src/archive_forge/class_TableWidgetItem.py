import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
class TableWidgetItem(QtWidgets.QTableWidgetItem):

    def __init__(self, val, index, format=None):
        QtWidgets.QTableWidgetItem.__init__(self, '')
        self._blockValueChange = False
        self._format = None
        self._defaultFormat = '%0.3g'
        self.sortMode = 'value'
        self.index = index
        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        self.setFlags(flags)
        self.setValue(val)
        self.setFormat(format)

    def setEditable(self, editable):
        """
        Set whether this item is user-editable.
        """
        if editable:
            self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        else:
            self.setFlags(self.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

    def setSortMode(self, mode):
        """
        Set the mode used to sort this item against others in its column.
        
        ============== ========================================================
        **Sort Modes**
        value          Compares item.value if available; falls back to text
                       comparison.
        text           Compares item.text()
        index          Compares by the order in which items were inserted.
        ============== ========================================================
        """
        modes = ('value', 'text', 'index', None)
        if mode not in modes:
            raise ValueError('Sort mode must be one of %s' % str(modes))
        self.sortMode = mode

    def setFormat(self, fmt):
        """Define the conversion from item value to displayed text. 
        
        If a string is specified, it is used as a format string for converting
        float values (and all other types are converted using str). If a 
        function is specified, it will be called with the item as its only
        argument and must return a string.
        
        Added in version 0.9.9.
        """
        if fmt is not None and (not isinstance(fmt, str)) and (not callable(fmt)):
            raise ValueError('Format argument must string, callable, or None. (got %s)' % fmt)
        self._format = fmt
        self._updateText()

    def _updateText(self):
        self._blockValueChange = True
        try:
            self._text = self.format()
            self.setText(self._text)
        finally:
            self._blockValueChange = False

    def setValue(self, value):
        self.value = value
        self._updateText()

    def itemChanged(self):
        """Called when the data of this item has changed."""
        if self.text() != self._text:
            self.textChanged()

    def textChanged(self):
        """Called when this item's text has changed for any reason."""
        self._text = self.text()
        if self._blockValueChange:
            return
        try:
            self.value = type(self.value)(self.text())
        except ValueError:
            self.value = str(self.text())

    def format(self):
        if callable(self._format):
            return self._format(self)
        if isinstance(self.value, (float, np.floating)):
            if self._format is None:
                return self._defaultFormat % self.value
            else:
                return self._format % self.value
        else:
            return str(self.value)

    def __lt__(self, other):
        if self.sortMode == 'index' and hasattr(other, 'index'):
            return self.index < other.index
        if self.sortMode == 'value' and hasattr(other, 'value'):
            return self.value < other.value
        else:
            return self.text() < other.text()