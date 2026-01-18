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
def test_stdioTransport(self):
    """
        The I{-bs} option for using stdin and stdout as the SMTP transport
        is unsupported and if it is passed to L{parseOptions}, L{SystemExit}
        is raised.
        """
    exc = self.assertRaises(SystemExit, parseOptions, ['-bs'])
    self.assertEqual(exc.args, ('Unsupported option.',))