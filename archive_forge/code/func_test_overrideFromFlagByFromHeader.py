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
def test_overrideFromFlagByFromHeader(self):
    """
        The I{-F} flag specifies the From: value.  However, I{-F} flag is
        overriden by the value of From: in the e-mail header.
        """
    stdin = StringIO('To: Curly <invaliduser4@example.com>\nFrom: Shemp <invaliduser4@example.com>\n')
    self.patch(sys, 'stdin', stdin)
    o = parseOptions(['-F', 'Groucho <invaliduser5@example.com>', '-t'])
    self.assertEqual(o.sender, 'invaliduser4@example.com')