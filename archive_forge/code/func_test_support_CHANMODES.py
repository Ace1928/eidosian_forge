import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_CHANMODES(self):
    """
        The CHANMODES ISUPPORT parameter is parsed into a C{dict} giving the
        four mode categories, C{'addressModes'}, C{'param'}, C{'setParam'}, and
        C{'noParam'}.
        """
    self._testFeatureDefault('CHANMODES')
    self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,')])
    self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,ha,ha')])
    self.assertEqual(self._parseFeature('CHANMODES', ',,,'), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
    self.assertEqual(self._parseFeature('CHANMODES', ',A,,'), {'addressModes': '', 'param': 'A', 'setParam': '', 'noParam': ''})
    self.assertEqual(self._parseFeature('CHANMODES', 'A,Bc,Def,Ghij'), {'addressModes': 'A', 'param': 'Bc', 'setParam': 'Def', 'noParam': 'Ghij'})