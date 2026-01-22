import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
class TestChannel(channel.SSHChannel):
    """
    A mocked-up version of twisted.conch.ssh.channel.SSHChannel.

    @ivar gotOpen: True if channelOpen has been called.
    @type gotOpen: L{bool}
    @ivar specificData: the specific channel open data passed to channelOpen.
    @type specificData: L{bytes}
    @ivar openFailureReason: the reason passed to openFailed.
    @type openFailed: C{error.ConchError}
    @ivar inBuffer: a C{list} of strings received by the channel.
    @type inBuffer: C{list}
    @ivar extBuffer: a C{list} of 2-tuples (type, extended data) of received by
        the channel.
    @type extBuffer: C{list}
    @ivar numberRequests: the number of requests that have been made to this
        channel.
    @type numberRequests: L{int}
    @ivar gotEOF: True if the other side sent EOF.
    @type gotEOF: L{bool}
    @ivar gotOneClose: True if the other side closed the connection.
    @type gotOneClose: L{bool}
    @ivar gotClosed: True if the channel is closed.
    @type gotClosed: L{bool}
    """
    name = b'TestChannel'
    gotOpen = False
    gotClosed = False

    def logPrefix(self):
        return 'TestChannel %i' % self.id

    def channelOpen(self, specificData):
        """
        The channel is open.  Set up the instance variables.
        """
        self.gotOpen = True
        self.specificData = specificData
        self.inBuffer = []
        self.extBuffer = []
        self.numberRequests = 0
        self.gotEOF = False
        self.gotOneClose = False
        self.gotClosed = False

    def openFailed(self, reason):
        """
        Opening the channel failed.  Store the reason why.
        """
        self.openFailureReason = reason

    def request_test(self, data):
        """
        A test request.  Return True if data is 'data'.

        @type data: L{bytes}
        """
        self.numberRequests += 1
        return data == b'data'

    def dataReceived(self, data):
        """
        Data was received.  Store it in the buffer.
        """
        self.inBuffer.append(data)

    def extReceived(self, code, data):
        """
        Extended data was received.  Store it in the buffer.
        """
        self.extBuffer.append((code, data))

    def eofReceived(self):
        """
        EOF was received.  Remember it.
        """
        self.gotEOF = True

    def closeReceived(self):
        """
        Close was received.  Remember it.
        """
        self.gotOneClose = True

    def closed(self):
        """
        The channel is closed.  Rembember it.
        """
        self.gotClosed = True