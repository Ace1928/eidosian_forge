import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class FakeUI:
    """
    A fake UI object, adhering to the interface expected by
    L{KnownHostsFile.verifyHostKey}

    @ivar userWarnings: inputs provided to 'warn'.

    @ivar promptDeferred: last result returned from 'prompt'.

    @ivar promptText: the last input provided to 'prompt'.
    """

    def __init__(self):
        self.userWarnings = []
        self.promptDeferred = None
        self.promptText = None

    def prompt(self, text):
        """
        Issue the user an interactive prompt, which they can accept or deny.
        """
        self.promptText = text
        self.promptDeferred = Deferred()
        return self.promptDeferred

    def warn(self, text):
        """
        Issue a non-interactive warning to the user.
        """
        self.userWarnings.append(text)