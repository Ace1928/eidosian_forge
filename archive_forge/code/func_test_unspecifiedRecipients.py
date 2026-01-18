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
def test_unspecifiedRecipients(self):
    """
        If no recipients are given in the argument list and there is no
        recipient header in the message text, L{parseOptions} raises
        L{SystemExit} with a string describing the problem.
        """
    self.patch(sys, 'stdin', StringIO('Subject: foo\n\nHello, goodbye.\n'))
    exc = self.assertRaises(SystemExit, parseOptions, [])
    self.assertEqual(exc.args, ('No recipients specified.',))