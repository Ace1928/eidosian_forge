import logging
import os
import sys
import warnings
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Generator, Optional, Set, Tuple, Union
import anyio
from ._rust_notify import RustNotify
from .filters import DefaultFilter
def _log_changes(changes: Set[FileChange]) -> None:
    if logger.isEnabledFor(logging.INFO):
        count = len(changes)
        plural = '' if count == 1 else 's'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('%d change%s detected: %s', count, plural, changes)
        else:
            logger.info('%d change%s detected', count, plural)