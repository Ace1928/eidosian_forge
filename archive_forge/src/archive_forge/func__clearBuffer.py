from twisted.protocols import loopback
def _clearBuffer(self):
    self.clearCall = None
    loopback.LoopbackRelay.clearBuffer(self)