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

        L{twisted.mail.scripts.mailmail.senderror} sends mail back to the
        sender if an error occurs while sending mail to the recipient.
        