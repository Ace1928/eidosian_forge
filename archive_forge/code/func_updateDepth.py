import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
def updateDepth(self, depth):
    """
        Change set the item font to bold and increase the font size on outermost groups.
        """
    for c in [0, 1]:
        font = self.font(c)
        font.setBold(True)
        if depth == 0:
            font.setPointSize(self.pointSize() + 1)
        self.setFont(c, font)
    self.titleChanged()