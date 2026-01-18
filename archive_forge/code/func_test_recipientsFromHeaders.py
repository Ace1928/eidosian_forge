import os
import sys
from io import StringIO
from unittest import skipIf
from twisted.copyright import version
from twisted.internet.defer import Deferred
from twisted.internet.testing import MemoryReactor
from twisted.mail import smtp
from twisted.mail.scripts import mailmail
from twisted.mail.scripts.mailmail import parseOptions
from twisted.python.failure import Failure
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
def test_recipientsFromHeaders(self):
    """
        The I{-t} flags specifies that recipients should be obtained
        from headers.
        """
    stdin = StringIO('To: Curly <invaliduser2@example.com>\nCc: Larry <invaliduser1@example.com>\nBcc: Moe <invaliduser3@example.com>\n\nOh, a wise guy?\n')
    self.patch(sys, 'stdin', stdin)
    o = parseOptions('-t')
    self.assertEqual(len(o.to), 3)