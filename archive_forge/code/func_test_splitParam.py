import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitParam(self):
    """
        L{ServerSupportedFeatures._splitParam} splits ISUPPORT parameters
        into key and values. Parameters without a separator are split into a
        key and a list containing only the empty string. Escaped parameters
        are unescaped.
        """
    params = [('FOO', ('FOO', [''])), ('FOO=', ('FOO', [''])), ('FOO=1', ('FOO', ['1'])), ('FOO=1,2,3', ('FOO', ['1', '2', '3'])), ('FOO=A\\x20B', ('FOO', ['A B'])), ('FOO=\\x5Cx', ('FOO', ['\\x'])), ('FOO=\\', ('FOO', ['\\'])), ('FOO=\\n', ('FOO', ['\\n']))]
    _splitParam = irc.ServerSupportedFeatures._splitParam
    for param, expected in params:
        res = _splitParam(param)
        self.assertEqual(res, expected)
    self.assertRaises(ValueError, _splitParam, 'FOO=\\x')
    self.assertRaises(ValueError, _splitParam, 'FOO=\\xNN')
    self.assertRaises(ValueError, _splitParam, 'FOO=\\xN')
    self.assertRaises(ValueError, _splitParam, 'FOO=\\x20\\x')