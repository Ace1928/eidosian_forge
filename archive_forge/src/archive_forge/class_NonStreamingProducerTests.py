from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
class NonStreamingProducerTests(TestCase):
    """
    Non-streaming producers can be adapted into being streaming producers.
    """

    def streamUntilEnd(self, consumer):
        """
        Verify the consumer writes out all its data, but is not called after
        that.
        """
        nsProducer = NonStreamingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        consumer.registerProducer(streamingProducer, True)

        def unregister(orig=consumer.unregisterProducer):
            orig()
            streamingProducer.stopStreaming()
        consumer.unregisterProducer = unregister
        done = nsProducer.result

        def doneStreaming(_):
            self.assertEqual(consumer.value(), b'0123456789')
            self.assertIsNone(consumer.producer)
            self.assertTrue(streamingProducer._finished)
        done.addCallback(doneStreaming)
        streamingProducer.startStreaming()
        return done

    def test_writeUntilDone(self):
        """
        When converted to a streaming producer, the non-streaming producer
        writes out all its data, but is not called after that.
        """
        consumer = StringTransport()
        return self.streamUntilEnd(consumer)

    def test_pause(self):
        """
        When the streaming producer is paused, the underlying producer stops
        getting resumeProducing calls.
        """

        class PausingStringTransport(StringTransport):
            writes = 0

            def __init__(self):
                StringTransport.__init__(self)
                self.paused = Deferred()

            def write(self, data):
                self.writes += 1
                StringTransport.write(self, data)
                if self.writes == 3:
                    self.producer.pauseProducing()
                    d = self.paused
                    del self.paused
                    d.callback(None)
        consumer = PausingStringTransport()
        nsProducer = NonStreamingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        consumer.registerProducer(streamingProducer, True)

        def shouldNotBeCalled(ignore):
            self.fail('BUG: The producer should not finish!')
        nsProducer.result.addCallback(shouldNotBeCalled)
        done = consumer.paused

        def paused(ignore):
            self.assertEqual(streamingProducer._coopTask._pauseCount, 1)
        done.addCallback(paused)
        streamingProducer.startStreaming()
        return done

    def test_resume(self):
        """
        When the streaming producer is paused and then resumed, the underlying
        producer starts getting resumeProducing calls again after the resume.

        The test will never finish (or rather, time out) if the resume
        producing call is not working.
        """

        class PausingStringTransport(StringTransport):
            writes = 0

            def write(self, data):
                self.writes += 1
                StringTransport.write(self, data)
                if self.writes == 3:
                    self.producer.pauseProducing()
                    self.producer.resumeProducing()
        consumer = PausingStringTransport()
        return self.streamUntilEnd(consumer)

    def test_stopProducing(self):
        """
        When the streaming producer is stopped by the consumer, the underlying
        producer is stopped, and streaming is stopped.
        """

        class StoppingStringTransport(StringTransport):
            writes = 0

            def write(self, data):
                self.writes += 1
                StringTransport.write(self, data)
                if self.writes == 3:
                    self.producer.stopProducing()
        consumer = StoppingStringTransport()
        nsProducer = NonStreamingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        consumer.registerProducer(streamingProducer, True)
        done = nsProducer.result

        def doneStreaming(_):
            self.assertEqual(consumer.value(), b'012')
            self.assertTrue(nsProducer.stopped)
            self.assertTrue(streamingProducer._finished)
        done.addCallback(doneStreaming)
        streamingProducer.startStreaming()
        return done

    def resumeProducingRaises(self, consumer, expectedExceptions):
        """
        Common implementation for tests where the underlying producer throws
        an exception when its resumeProducing is called.
        """

        class ThrowingProducer(NonStreamingProducer):

            def resumeProducing(self):
                if self.counter == 2:
                    return 1 / 0
                else:
                    NonStreamingProducer.resumeProducing(self)
        nsProducer = ThrowingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        consumer.registerProducer(streamingProducer, True)
        loggedMsgs = []
        log.addObserver(loggedMsgs.append)
        self.addCleanup(log.removeObserver, loggedMsgs.append)

        def unregister(orig=consumer.unregisterProducer):
            orig()
            streamingProducer.stopStreaming()
        consumer.unregisterProducer = unregister
        streamingProducer.startStreaming()
        done = streamingProducer._coopTask.whenDone()
        done.addErrback(lambda reason: reason.trap(TaskStopped))

        def stopped(ign):
            self.assertEqual(consumer.value(), b'01')
            errors = self.flushLoggedErrors()
            self.assertEqual(len(errors), len(expectedExceptions))
            for f, (expected, msg), logMsg in zip(errors, expectedExceptions, loggedMsgs):
                self.assertTrue(f.check(expected))
                self.assertIn(msg, logMsg['why'])
            self.assertTrue(streamingProducer._finished)
        done.addCallback(stopped)
        return done

    def test_resumeProducingRaises(self):
        """
        If the underlying producer raises an exception when resumeProducing is
        called, the streaming wrapper should log the error, unregister from
        the consumer and stop streaming.
        """
        consumer = StringTransport()
        done = self.resumeProducingRaises(consumer, [(ZeroDivisionError, 'failed, producing will be stopped')])

        def cleanShutdown(ignore):
            self.assertIsNone(consumer.producer)
        done.addCallback(cleanShutdown)
        return done

    def test_resumeProducingRaiseAndUnregisterProducerRaises(self):
        """
        If the underlying producer raises an exception when resumeProducing is
        called, the streaming wrapper should log the error, unregister from
        the consumer and stop streaming even if the unregisterProducer call
        also raise.
        """
        consumer = StringTransport()

        def raiser():
            raise RuntimeError()
        consumer.unregisterProducer = raiser
        return self.resumeProducingRaises(consumer, [(ZeroDivisionError, 'failed, producing will be stopped'), (RuntimeError, 'failed to unregister producer')])

    def test_stopStreamingTwice(self):
        """
        stopStreaming() can be called more than once without blowing
        up. This is useful for error-handling paths.
        """
        consumer = StringTransport()
        nsProducer = NonStreamingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        streamingProducer.startStreaming()
        streamingProducer.stopStreaming()
        streamingProducer.stopStreaming()
        self.assertTrue(streamingProducer._finished)

    def test_interface(self):
        """
        L{_PullToPush} implements L{IPushProducer}.
        """
        consumer = StringTransport()
        nsProducer = NonStreamingProducer(consumer)
        streamingProducer = _PullToPush(nsProducer, consumer)
        self.assertTrue(verifyObject(IPushProducer, streamingProducer))