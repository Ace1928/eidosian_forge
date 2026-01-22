from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakValueDictionary
from ._error import Timeout
@dataclass
class FileLockContext:
    """A dataclass which holds the context for a ``BaseFileLock`` object."""
    lock_file: str
    timeout: float
    mode: int
    lock_file_fd: int | None = None
    lock_counter: int = 0