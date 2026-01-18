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
def test_senderror(self):
    """
        L{twisted.mail.scripts.mailmail.senderror} sends mail back to the
        sender if an error occurs while sending mail to the recipient.
        """

    def sendmail(host, sender, recipient, body):
        self.assertRegex(sender, 'postmaster@')
        self.assertEqual(recipient, ['testsender'])
        self.assertRegex(body.getvalue(), 'ValueError')
        return Deferred()
    self.patch(smtp, 'sendmail', sendmail)
    opts = mailmail.Options()
    opts.sender = 'testsender'
    fail = Failure(ValueError())
    mailmail.senderror(fail, opts)