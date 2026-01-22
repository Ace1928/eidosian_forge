from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
@implementer(_IFileDescriptorReservation)
@attr.s(auto_attribs=True)
class _FileDescriptorReservation:
    """
    L{_IFileDescriptorReservation} implementation.

    @ivar fileFactory: A factory that will be called to reserve a
        file descriptor.
    @type fileFactory: A L{callable} that accepts no arguments and
        returns an object with a C{close} method.
    """
    _log: ClassVar[Logger] = Logger()
    _fileFactory: Callable[[], _HasClose]
    _fileDescriptor: Optional[_HasClose] = attr.ib(init=False, default=None)

    def available(self):
        """
        See L{_IFileDescriptorReservation.available}.

        @return: L{True} if the reserved file descriptor is open and
            can thus be closed to allow a new file to be opened in its
            place; L{False} if it is not open.
        """
        return self._fileDescriptor is not None

    def reserve(self):
        """
        See L{_IFileDescriptorReservation.reserve}.
        """
        if self._fileDescriptor is None:
            try:
                fileDescriptor = self._fileFactory()
            except OSError as e:
                if e.errno == EMFILE:
                    self._log.failure('Could not reserve EMFILE recovery file descriptor.')
                else:
                    raise
            else:
                self._fileDescriptor = fileDescriptor

    def __enter__(self):
        """
        See L{_IFileDescriptorReservation.__enter__}.
        """
        if self._fileDescriptor is None:
            raise RuntimeError('No file reserved.  Have you called my reserve method?')
        self._fileDescriptor.close()
        self._fileDescriptor = None

    def __exit__(self, excType, excValue, traceback):
        """
        See L{_IFileDescriptorReservation.__exit__}.
        """
        try:
            self.reserve()
        except Exception:
            self._log.failure('Could not re-reserve EMFILE recovery file descriptor.')