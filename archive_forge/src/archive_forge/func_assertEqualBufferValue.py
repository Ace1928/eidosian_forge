import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def assertEqualBufferValue(self, buf, val):
    """
        A buffer is always bytes, but sometimes
        we need to compare it to a utf-8 unicode string

        @param buf: the buffer
        @type buf: L{bytes} or L{unicode} or L{list}
        @param val: the value to compare
        @type val: L{bytes} or L{unicode} or L{list}
        """
    bufferValue = buf
    if isinstance(val, str):
        bufferValue = bufferValue.decode('utf-8')
    if isinstance(bufferValue, list):
        if isinstance(val[0], str):
            bufferValue = [b.decode('utf8') for b in bufferValue]
    self.assertEqual(bufferValue, val)