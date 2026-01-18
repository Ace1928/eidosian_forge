from zope.interface import implementer
from twisted.internet import interfaces
from twisted.logger import Logger
from twisted.python import log
def writeExtended(self, dataType, data):
    """
        Send extended data to this channel.  If there is not enough remote
        window available, buffer until there is.  Otherwise, split the data
        into packets of length remoteMaxPacket and send them.

        @type dataType: L{int}
        @type data:     L{bytes}
        """
    if self.extBuf:
        if self.extBuf[-1][0] == dataType:
            self.extBuf[-1][1] += data
        else:
            self.extBuf.append([dataType, data])
        return
    if len(data) > self.remoteWindowLeft:
        data, self.extBuf = (data[:self.remoteWindowLeft], [[dataType, data[self.remoteWindowLeft:]]])
        self.areWriting = 0
        self.stopWriting()
    while len(data) > self.remoteMaxPacket:
        self.conn.sendExtendedData(self, dataType, data[:self.remoteMaxPacket])
        data = data[self.remoteMaxPacket:]
        self.remoteWindowLeft -= self.remoteMaxPacket
    if data:
        self.conn.sendExtendedData(self, dataType, data)
        self.remoteWindowLeft -= len(data)
    if self.closing:
        self.loseConnection()