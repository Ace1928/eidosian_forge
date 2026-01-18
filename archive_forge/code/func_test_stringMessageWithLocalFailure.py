import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_stringMessageWithLocalFailure(self):
    """
        L{amp.RemoteAmpError} renders local errors with a "(local)" marker and
        a brief traceback.
        """
    failure = Failure(Exception('Something came loose'))
    error = amp.RemoteAmpError(b'BROKEN', 'Something has broken', local=failure)
    self.assertRegex(str(error), '^Code<BROKEN> [(]local[)]: Something has broken\nTraceback [(]failure with no frames[)]: <.+Exception.>: Something came loose\n')