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
@implementer(_IExhaustsFileDescriptors)
@attr.s(auto_attribs=True)
class _ExhaustsFileDescriptors:
    """
    A class that triggers C{EMFILE} by creating as many file
    descriptors as necessary.

    @ivar fileDescriptorFactory: A factory that creates a new file
        descriptor.
    @type fileDescriptorFactory: A L{callable} that accepts no
        arguments and returns an integral file descriptor, suitable
        for passing to L{os.close}.
    """
    _log: ClassVar[Logger] = Logger()
    _fileDescriptorFactory: Callable[[], int] = attr.ib(default=lambda: os.dup(0), repr=False)
    _close: Callable[[int], None] = attr.ib(default=os.close, repr=False)
    _fileDescriptors: List[int] = attr.ib(default=attr.Factory(list), init=False, repr=False)

    def exhaust(self):
        """
        Open file descriptors until C{EMFILE} is reached.
        """
        gc.collect()
        try:
            while True:
                try:
                    fd = self._fileDescriptorFactory()
                except OSError as e:
                    if e.errno == errno.EMFILE:
                        break
                    raise
                else:
                    self._fileDescriptors.append(fd)
        except Exception:
            self.release()
            raise
        else:
            self._log.info('EMFILE reached by opening {openedFileDescriptors} file descriptors.', openedFileDescriptors=self.count())

    def release(self):
        """
        Release all file descriptors opened by L{exhaust}.
        """
        while self._fileDescriptors:
            fd = self._fileDescriptors.pop()
            try:
                self._close(fd)
            except OSError as e:
                if e.errno == errno.EBADF:
                    continue
                raise

    def count(self):
        """
        Return the number of opened file descriptors.

        @return: The number of opened file descriptors; this will be
            zero if this instance has not opened any.
        @rtype: L{int}
        """
        return len(self._fileDescriptors)