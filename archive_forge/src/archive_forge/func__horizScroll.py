import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def _horizScroll(self, n):
    self._viewport.xOffset += n
    self._viewport.xOffset = max(0, self._viewport.xOffset)
    return self._viewport.xOffset / 25.0