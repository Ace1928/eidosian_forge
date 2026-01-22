from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def addWindowBytes(self, data):
    """
        Called when bytes are added to the remote window.  By default it clears
        the data buffers.

        @type data:    L{bytes}
        """
    self.remoteWindowLeft = self.remoteWindowLeft + data
    if not self.areWriting and (not self.closing):
        self.areWriting = True
        self.startWriting()
    if self.buf:
        b = self.buf
        self.buf = b''
        self.write(b)
    if self.extBuf:
        b = self.extBuf
        self.extBuf = []
        for type, data in b:
            self.writeExtended(type, data)