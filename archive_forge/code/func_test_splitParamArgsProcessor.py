import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitParamArgsProcessor(self):
    """
        L{ServerSupportedFeatures._splitParamArgs} uses the argument processor
        passed to convert ISUPPORT argument values to some more suitable
        form.
        """
    res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C'], irc._intOrDefault)
    self.assertEqual(res, [('A', 1), ('B', 2), ('C', None)])