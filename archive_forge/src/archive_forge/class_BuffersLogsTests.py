import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class BuffersLogsTests(SynchronousTestCase):
    """
    Tests for L{_BuffersLogs}.
    """

    def setUp(self):
        self.namespace = 'name.space'
        self.events = []
        self.logBuffer = _BuffersLogs(self.namespace, self.events.append)

    def test_buffersInBlock(self):
        """
        The context manager's logger does not log to provided observer
        inside the block.
        """
        with self.logBuffer as logger:
            logger.info('An event')
            self.assertFalse(self.events)

    def test_flushesOnExit(self):
        """
        The context manager flushes its buffered logs when the block
        terminates without an exception.
        """
        with self.logBuffer as logger:
            logger.info('An event')
            self.assertFalse(self.events)
        self.assertEqual(1, len(self.events))
        [event] = self.events
        self.assertEqual(event['log_format'], 'An event')
        self.assertEqual(event['log_namespace'], self.namespace)

    def test_flushesOnExitWithException(self):
        """
        The context manager flushes its buffered logs when the block
        terminates because of an exception.
        """

        class TestException(Exception):
            """
            An exception only raised by this test.
            """
        with self.assertRaises(TestException):
            with self.logBuffer as logger:
                logger.info('An event')
                self.assertFalse(self.events)
                raise TestException()
        self.assertEqual(1, len(self.events))
        [event] = self.events
        self.assertEqual(event['log_format'], 'An event')
        self.assertEqual(event['log_namespace'], self.namespace)