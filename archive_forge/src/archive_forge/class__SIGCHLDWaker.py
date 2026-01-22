from __future__ import annotations
import contextlib
import errno
import os
import signal
import socket
from types import FrameType
from typing import Callable, Optional, Sequence
from zope.interface import Attribute, Interface, implementer
from attrs import define, frozen
from typing_extensions import Protocol, TypeAlias
from twisted.internet.interfaces import IReadDescriptor
from twisted.python import failure, log, util
from twisted.python.runtime import platformType
class _SIGCHLDWaker(_FDWaker):
    """
    L{_SIGCHLDWaker} can wake up a reactor whenever C{SIGCHLD} is received.
    """

    def install(self) -> None:
        """
        Install the handler necessary to make this waker active.
        """
        installHandler(self.o)

    def uninstall(self) -> None:
        """
        Remove the handler which makes this waker active.
        """
        installHandler(-1)

    def doRead(self) -> None:
        """
        Having woken up the reactor in response to receipt of
        C{SIGCHLD}, reap the process which exited.

        This is called whenever the reactor notices the waker pipe is
        writeable, which happens soon after any call to the C{wakeUp}
        method.
        """
        super().doRead()
        process.reapAllProcesses()