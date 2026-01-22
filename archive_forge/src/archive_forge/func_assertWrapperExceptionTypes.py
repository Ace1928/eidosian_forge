from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def assertWrapperExceptionTypes(self, deferred, mainType, reasonTypes):
    """
    Assert that the given L{Deferred} fails with the exception given by
    C{mainType} and that the exceptions wrapped by the instance of C{mainType}
    it fails with match the list of exception types given by C{reasonTypes}.

    This is a helper for testing failures of exceptions which subclass
    L{_newclient._WrapperException}.

    @param self: A L{TestCase} instance which will be used to make the
        assertions.

    @param deferred: The L{Deferred} which is expected to fail with
        C{mainType}.

    @param mainType: A L{_newclient._WrapperException} subclass which will be
        trapped on C{deferred}.

    @param reasonTypes: A sequence of exception types which will be trapped on
        the resulting C{mainType} exception instance's C{reasons} sequence.

    @return: A L{Deferred} which fires with the C{mainType} instance
        C{deferred} fails with, or which fails somehow.
    """

    def cbFailed(err):
        for reason, type in zip(err.reasons, reasonTypes):
            reason.trap(type)
        self.assertEqual(len(err.reasons), len(reasonTypes), f'len({err.reasons}) != len({reasonTypes})')
        return err
    d = self.assertFailure(deferred, mainType)
    d.addCallback(cbFailed)
    return d