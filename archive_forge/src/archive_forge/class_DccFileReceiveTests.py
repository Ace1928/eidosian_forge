import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class DccFileReceiveTests(IRCTestCase):
    """
    Tests for L{DccFileReceive}.
    """

    def makeConnectedDccFileReceive(self, filename, resumeOffset=0, overwrite=None):
        """
        Factory helper that returns a L{DccFileReceive} instance
        for a specific test case.

        @param filename: Path to the local file where received data is stored.
        @type filename: L{str}

        @param resumeOffset: An integer representing the amount of bytes from
            where the transfer of data should be resumed.
        @type resumeOffset: L{int}

        @param overwrite: A boolean specifying whether the file to write to
            should be overwritten by calling L{DccFileReceive.set_overwrite}
            or not.
        @type overwrite: L{bool}

        @return: An instance of L{DccFileReceive}.
        @rtype: L{DccFileReceive}
        """
        protocol = irc.DccFileReceive(filename, resumeOffset=resumeOffset)
        if overwrite:
            protocol.set_overwrite(True)
        transport = StringTransport()
        protocol.makeConnection(transport)
        return protocol

    def allDataReceivedForProtocol(self, protocol, data):
        """
        Arrange the protocol so that it received all data.

        @param protocol: The protocol which will receive the data.
        @type: L{DccFileReceive}

        @param data: The received data.
        @type data: L{bytest}
        """
        protocol.dataReceived(data)
        protocol.connectionLost(None)

    def test_resumeFromResumeOffset(self):
        """
        If given a resumeOffset argument, L{DccFileReceive} will attempt to
        resume from that number of bytes if the file exists.
        """
        fp = FilePath(self.mktemp())
        fp.setContent(b'Twisted is awesome!')
        protocol = self.makeConnectedDccFileReceive(fp.path, resumeOffset=11)
        self.allDataReceivedForProtocol(protocol, b'amazing!')
        self.assertEqual(fp.getContent(), b'Twisted is amazing!')

    def test_resumeFromResumeOffsetInTheMiddleOfAlreadyWrittenData(self):
        """
        When resuming from an offset somewhere in the middle of the file,
        for example, if there are 50 bytes in a file, and L{DccFileReceive}
        is given a resumeOffset of 25, and after that 15 more bytes are
        written to the file, then the resultant file should have just 40
        bytes of data.
        """
        fp = FilePath(self.mktemp())
        fp.setContent(b'Twisted is amazing!')
        protocol = self.makeConnectedDccFileReceive(fp.path, resumeOffset=11)
        self.allDataReceivedForProtocol(protocol, b'cool!')
        self.assertEqual(fp.getContent(), b'Twisted is cool!')

    def test_setOverwrite(self):
        """
        When local file already exists it can be overwritten using the
        L{DccFileReceive.set_overwrite} method.
        """
        fp = FilePath(self.mktemp())
        fp.setContent(b'I love contributing to Twisted!')
        protocol = self.makeConnectedDccFileReceive(fp.path, overwrite=True)
        self.allDataReceivedForProtocol(protocol, b'Twisted rocks!')
        self.assertEqual(fp.getContent(), b'Twisted rocks!')

    def test_fileDoesNotExist(self):
        """
        If the file does not already exist, then L{DccFileReceive} will
        create one and write the data to it.
        """
        fp = FilePath(self.mktemp())
        protocol = self.makeConnectedDccFileReceive(fp.path)
        self.allDataReceivedForProtocol(protocol, b'I <3 Twisted')
        self.assertEqual(fp.getContent(), b'I <3 Twisted')

    def test_resumeWhenFileDoesNotExist(self):
        """
        If given a resumeOffset to resume writing to a file that does not
        exist, L{DccFileReceive} will raise L{OSError}.
        """
        fp = FilePath(self.mktemp())
        error = self.assertRaises(OSError, self.makeConnectedDccFileReceive, fp.path, resumeOffset=1)
        self.assertEqual(errno.ENOENT, error.errno)

    def test_fileAlreadyExistsNoOverwrite(self):
        """
        If the file already exists and overwrite action was not asked,
        L{OSError} is raised.
        """
        fp = FilePath(self.mktemp())
        fp.touch()
        self.assertRaises(OSError, self.makeConnectedDccFileReceive, fp.path)

    def test_failToOpenLocalFile(self):
        """
        L{IOError} is raised when failing to open the requested path.
        """
        fp = FilePath(self.mktemp()).child('child-with-no-existing-parent')
        self.assertRaises(IOError, self.makeConnectedDccFileReceive, fp.path)