from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
class FontLayout(QtWidgets.QGridLayout):
    """Font selection"""

    def __init__(self, value, parent=None):
        super().__init__()
        font = tuple_to_qfont(value)
        assert font is not None
        self.family = QtWidgets.QFontComboBox(parent)
        self.family.setCurrentFont(font)
        self.addWidget(self.family, 0, 0, 1, -1)
        self.size = QtWidgets.QComboBox(parent)
        self.size.setEditable(True)
        sizelist = [*range(6, 12), *range(12, 30, 2), 36, 48, 72]
        size = font.pointSize()
        if size not in sizelist:
            sizelist.append(size)
            sizelist.sort()
        self.size.addItems([str(s) for s in sizelist])
        self.size.setCurrentIndex(sizelist.index(size))
        self.addWidget(self.size, 1, 0)
        self.italic = QtWidgets.QCheckBox(self.tr('Italic'), parent)
        self.italic.setChecked(font.italic())
        self.addWidget(self.italic, 1, 1)
        self.bold = QtWidgets.QCheckBox(self.tr('Bold'), parent)
        self.bold.setChecked(font.bold())
        self.addWidget(self.bold, 1, 2)

    def get_font(self):
        font = self.family.currentFont()
        font.setItalic(self.italic.isChecked())
        font.setBold(self.bold.isChecked())
        font.setPointSize(int(self.size.currentText()))
        return qfont_to_tuple(font)