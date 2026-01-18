import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_parseChanModesParam(self):
    """
        L{ServerSupportedFeatures._parseChanModesParam} parses the ISUPPORT
        CHANMODES parameter into a mapping from mode categories to mode
        characters. Passing fewer than 4 parameters results in the empty string
        for the relevant categories. Passing more than 4 parameters raises
        C{ValueError}.
        """
    _parseChanModesParam = irc.ServerSupportedFeatures._parseChanModesParam
    self.assertEqual(_parseChanModesParam(['', '', '', '']), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
    self.assertEqual(_parseChanModesParam(['b', 'k', 'l', 'imnpst']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': 'imnpst'})
    self.assertEqual(_parseChanModesParam(['b', 'k', 'l', '']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': ''})
    self.assertRaises(ValueError, _parseChanModesParam, ['a', 'b', 'c', 'd', 'e'])