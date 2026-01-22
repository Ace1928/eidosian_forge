import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class _Scrollbar(Widget):

    def __init__(self, onScroll):
        Widget.__init__(self)
        self.onScroll = onScroll
        self.percent = 0.0

    def smaller(self):
        self.percent = min(1.0, max(0.0, self.onScroll(-1)))
        self.repaint()

    def bigger(self):
        self.percent = min(1.0, max(0.0, self.onScroll(+1)))
        self.repaint()