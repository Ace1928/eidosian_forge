import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def updateStyle(self):
    r = '3px'
    if self.dim:
        fg = '#aaa'
        bg = '#44a'
        border = '#339'
    else:
        fg = '#fff'
        bg = '#66c'
        border = '#55B'
    if self.orientation == 'vertical':
        self.vStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: 0px;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: %s;\n                border-width: 0px;\n                border-right: 2px solid %s;\n                padding-top: 3px;\n                padding-bottom: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
        self.setStyleSheet(self.vStyle)
    else:
        self.hStyle = 'DockLabel {\n                background-color : %s;\n                color : %s;\n                border-top-right-radius: %s;\n                border-top-left-radius: %s;\n                border-bottom-right-radius: 0px;\n                border-bottom-left-radius: 0px;\n                border-width: 0px;\n                border-bottom: 2px solid %s;\n                padding-left: 3px;\n                padding-right: 3px;\n                font-size: %s;\n            }' % (bg, fg, r, r, border, self.fontSize)
        self.setStyleSheet(self.hStyle)