from __future__ import annotations
import functools
import logging
import sys
import typing
from twisted.internet.abstract import FileDescriptor
from twisted.internet.error import AlreadyCalled, AlreadyCancelled
from .abstract_loop import EventLoop, ExitMainLoop

        Decorator that cleanly exits the :class:`TwistedEventLoop` if
        :class:`ExitMainLoop` is thrown inside of the wrapped function. Store the
        exception info if some other exception occurs, it will be reraised after
        the loop quits.

        *f* -- function to be wrapped
        