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
@frozen
class _WithoutSignalHandling:
    """
    A L{SignalHandling} implementation that does no signal handling.

    This is the implementation of C{installSignalHandlers=False}.
    """

    def install(self) -> None:
        """
        Do not install any signal handlers.
        """

    def uninstall(self) -> None:
        """
        Do nothing because L{install} installed nothing.
        """