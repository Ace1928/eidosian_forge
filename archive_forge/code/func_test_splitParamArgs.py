import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitParamArgs(self):
    """
        L{ServerSupportedFeatures._splitParamArgs} splits ISUPPORT parameter
        arguments into key and value.  Arguments without a separator are
        split into a key and an empty string.
        """
    res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C:', 'D'])
    self.assertEqual(res, [('A', '1'), ('B', '2'), ('C', ''), ('D', '')])