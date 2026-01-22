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
@skipIf(SKIP_EMFILE, 'Reserved EMFILE file descriptor not supported on Windows.')
class FileDescriptorReservationTests(SynchronousTestCase):
    """
    Tests for L{_FileDescriptorReservation}.
    """

    def setUp(self):
        self.reservedFileObjects = []
        self.tempfile = self.mktemp()

        def fakeFileFactory():
            self.reservedFileObjects.append(open(self.tempfile, 'w'))
            return self.reservedFileObjects[-1]
        self.reservedFD = _FileDescriptorReservation(fakeFileFactory)

    def test_providesInterface(self):
        """
        L{_FileDescriptorReservation} instances provide
        L{_IFileDescriptorReservation}.
        """
        verifyObject(_IFileDescriptorReservation, self.reservedFD)

    def test_reserveOpensFileOnce(self):
        """
        Multiple acquisitions without releases open the reservation
        file exactly once.
        """
        self.assertEqual(len(self.reservedFileObjects), 0)
        for _ in range(10):
            self.reservedFD.reserve()
            self.assertEqual(len(self.reservedFileObjects), 1)
            self.assertFalse(self.reservedFileObjects[0].closed)

    def test_reserveEMFILELogged(self):
        """
        If reserving the file descriptor fails because of C{EMFILE},
        the exception is suppressed but logged and the reservation
        remains unavailable.
        """
        exhauster = _ExhaustsFileDescriptors()
        self.addCleanup(exhauster.release)
        exhauster.exhaust()
        self.assertFalse(self.reservedFD.available())
        self.reservedFD.reserve()
        self.assertFalse(self.reservedFD.available())
        errors = self.flushLoggedErrors(OSError, IOError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.errno, errno.EMFILE)

    def test_reserveRaisesNonEMFILEExceptions(self):
        """
        Any exception raised while opening the reserve file that is
        not an L{OSError} or L{IOError} whose errno is C{EMFILE} is
        allowed through to the caller.
        """
        for errorClass in (OSError, IOError, ValueError):

            def failsWith(errorClass=errorClass):
                raise errorClass(errno.EMFILE + 1, 'message')
            reserveFD = _FileDescriptorReservation(failsWith)
            self.assertRaises(errorClass, reserveFD.reserve)

    def test_available(self):
        """
        The reservation is available after the file descriptor is
        reserved.
        """
        self.assertFalse(self.reservedFD.available())
        self.reservedFD.reserve()
        self.assertTrue(self.reservedFD.available())

    def test_enterFailsWithoutFile(self):
        """
        A reservation without an open file used as a context manager
        raises a L{RuntimeError}.
        """
        with self.assertRaises(RuntimeError):
            with self.reservedFD:
                'This string cannot raise an exception.'

    def test_enterClosesFileExitOpensFile(self):
        """
        Entering a reservation closes its file for the duration of the
        context manager's block.
        """
        self.reservedFD.reserve()
        self.assertTrue(self.reservedFD.available())
        with self.reservedFD:
            self.assertFalse(self.reservedFD.available())
        self.assertTrue(self.reservedFD.available())

    def test_exitOpensFileOnException(self):
        """
        An exception raised within a reservation context manager's
        block does not prevent the file from being reopened.
        """

        class TestException(Exception):
            """
            An exception only used by this test.
            """
        self.reservedFD.reserve()
        with self.assertRaises(TestException):
            with self.reservedFD:
                raise TestException()

    def test_exitSuppressesReservationException(self):
        """
        An exception raised while re-opening the reserve file exiting
        a reservation's context manager block is suppressed but
        logged, allowing an exception raised within the block through.
        """

        class AllowedException(Exception):
            """
            The exception allowed out of the block.
            """

        class SuppressedException(Exception):
            """
            An exception raised by the file descriptor factory.
            """
        called = [False]

        def failsWithSuppressedExceptionAfterSecondOpen():
            if called[0]:
                raise SuppressedException()
            else:
                called[0] = True
                return io.BytesIO()
        reservedFD = _FileDescriptorReservation(failsWithSuppressedExceptionAfterSecondOpen)
        reservedFD.reserve()
        self.assertTrue(reservedFD.available())
        with self.assertRaises(AllowedException):
            with reservedFD:
                raise AllowedException()
        errors = self.flushLoggedErrors(SuppressedException)
        self.assertEqual(len(errors), 1)