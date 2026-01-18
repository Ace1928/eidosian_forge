import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
@xOffset.setter
def xOffset(self, value):
    if self._xOffset != value:
        self._xOffset = value
        self.repaint()