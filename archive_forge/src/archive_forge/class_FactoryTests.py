from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
class FactoryTests(TestCase):
    """
    Tests for L{protocol.Factory}.
    """

    def test_interfaces(self):
        """
        L{Factory} instances provide both L{IProtocolFactory} and
        L{ILoggingContext}.
        """
        factory = Factory()
        self.assertTrue(verifyObject(IProtocolFactory, factory))
        self.assertTrue(verifyObject(ILoggingContext, factory))

    def test_logPrefix(self):
        """
        L{Factory.logPrefix} returns the name of the factory class.
        """

        class SomeKindOfFactory(Factory):
            pass
        self.assertEqual('SomeKindOfFactory', SomeKindOfFactory().logPrefix())

    def test_defaultBuildProtocol(self):
        """
        L{Factory.buildProtocol} by default constructs a protocol by calling
        its C{protocol} attribute, and attaches the factory to the result.
        """

        class SomeProtocol(Protocol):
            pass
        f = Factory()
        f.protocol = SomeProtocol
        protocol = f.buildProtocol(None)
        self.assertIsInstance(protocol, SomeProtocol)
        self.assertIs(protocol.factory, f)

    def test_forProtocol(self):
        """
        L{Factory.forProtocol} constructs a Factory, passing along any
        additional arguments, and sets its C{protocol} attribute to the given
        Protocol subclass.
        """

        class ArgTakingFactory(Factory):

            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = (args, kwargs)
        factory = ArgTakingFactory.forProtocol(Protocol, 1, 2, foo=12)
        self.assertEqual(factory.protocol, Protocol)
        self.assertEqual(factory.args, (1, 2))
        self.assertEqual(factory.kwargs, {'foo': 12})

    def test_doStartLoggingStatement(self):
        """
        L{Factory.doStart} logs that it is starting a factory, followed by
        the L{repr} of the L{Factory} instance that is being started.
        """
        events = []
        globalLogPublisher.addObserver(events.append)
        self.addCleanup(lambda: globalLogPublisher.removeObserver(events.append))
        f = Factory()
        f.doStart()
        self.assertIs(events[0]['factory'], f)
        self.assertEqual(events[0]['log_level'], LogLevel.info)
        self.assertEqual(events[0]['log_format'], 'Starting factory {factory!r}')

    def test_doStopLoggingStatement(self):
        """
        L{Factory.doStop} logs that it is stopping a factory, followed by
        the L{repr} of the L{Factory} instance that is being stopped.
        """
        events = []
        globalLogPublisher.addObserver(events.append)
        self.addCleanup(lambda: globalLogPublisher.removeObserver(events.append))

        class MyFactory(Factory):
            numPorts = 1
        f = MyFactory()
        f.doStop()
        self.assertIs(events[0]['factory'], f)
        self.assertEqual(events[0]['log_level'], LogLevel.info)
        self.assertEqual(events[0]['log_format'], 'Stopping factory {factory!r}')