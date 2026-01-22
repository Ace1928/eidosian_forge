from __future__ import annotations
import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from .util import py311
from .util import py38
from .util.typing import Literal
class InstanceLogger:
    """A logger adapter (wrapper) for :class:`.Identified` subclasses.

    This allows multiple instances (e.g. Engine or Pool instances)
    to share a logger, but have its verbosity controlled on a
    per-instance basis.

    The basic functionality is to return a logging level
    which is based on an instance's echo setting.

    Default implementation is:

    'debug' -> logging.DEBUG
    True    -> logging.INFO
    False   -> Effective level of underlying logger (
    logging.WARNING by default)
    None    -> same as False
    """
    _echo_map = {None: logging.NOTSET, False: logging.NOTSET, True: logging.INFO, 'debug': logging.DEBUG}
    _echo: _EchoFlagType
    __slots__ = ('echo', 'logger')

    def __init__(self, echo: _EchoFlagType, name: str):
        self.echo = echo
        self.logger = logging.getLogger(name)
        if self._echo_map[echo] <= logging.INFO and (not self.logger.handlers):
            _add_default_handler(self.logger)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a debug call to the underlying logger."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an info call to the underlying logger."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a warning call to the underlying logger."""
        self.log(logging.WARNING, msg, *args, **kwargs)
    warn = warning

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate an error call to the underlying logger.
        """
        self.log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an exception call to the underlying logger."""
        kwargs['exc_info'] = 1
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a critical call to the underlying logger."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a log call to the underlying logger.

        The level here is determined by the echo
        flag as well as that of the underlying logger, and
        logger._log() is called directly.

        """
        if self.logger.manager.disable >= level:
            return
        selected_level = self._echo_map[self.echo]
        if selected_level == logging.NOTSET:
            selected_level = self.logger.getEffectiveLevel()
        if level >= selected_level:
            if STACKLEVEL:
                kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + STACKLEVEL_OFFSET
            self.logger._log(level, msg, args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        """Is this logger enabled for level 'level'?"""
        if self.logger.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getEffectiveLevel(self) -> int:
        """What's the effective level for this logger?"""
        level = self._echo_map[self.echo]
        if level == logging.NOTSET:
            level = self.logger.getEffectiveLevel()
        return level