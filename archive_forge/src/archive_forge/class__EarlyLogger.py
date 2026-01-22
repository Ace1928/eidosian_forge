import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
class _EarlyLogger:
    """Early logger which captures logs in memory until logging can be configured."""

    def __init__(self) -> None:
        self._log: List[tuple] = []
        self._exception: List[tuple] = []
        self.warn = self.warning

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((logging.DEBUG, msg, args, kwargs))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((logging.INFO, msg, args, kwargs))

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((logging.WARNING, msg, args, kwargs))

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((logging.ERROR, msg, args, kwargs))

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((logging.CRITICAL, msg, args, kwargs))

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._exception.append((msg, args, kwargs))

    def log(self, level: str, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log.append((level, msg, args, kwargs))

    def _flush(self) -> None:
        assert self is not logger
        assert logger is not None
        for level, msg, args, kwargs in self._log:
            logger.log(level, msg, *args, **kwargs)
        for msg, args, kwargs in self._exception:
            logger.exception(msg, *args, **kwargs)