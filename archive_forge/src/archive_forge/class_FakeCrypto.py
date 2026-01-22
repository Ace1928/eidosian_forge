import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class FakeCrypto:
    """
    An introspectable fake of pyOpenSSL's L{OpenSSL.crypto} module.

    @ivar state: A L{FakeCryptoState} instance
    """

    def __init__(self, state):
        self._state = state

    def get_elliptic_curve(self, curve):
        """
        A fake that records the curve with which it was called.

        @param curve: see L{crypto.get_elliptic_curve}

        @return: see L{FakeCryptoState.getEllipticCurveReturns}
        @raises: see L{FakeCryptoState.getEllipticCurveRaises}
        """
        self._state.getEllipticCurveCalls.append(curve)
        if self._state.getEllipticCurveRaises is not None:
            raise self._state.getEllipticCurveRaises
        return self._state.getEllipticCurveReturns