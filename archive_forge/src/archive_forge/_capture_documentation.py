from contextlib import contextmanager
from typing import Iterator, List, Sequence, cast
from twisted.logger import globalLogPublisher
from ._interfaces import ILogObserver, LogEvent

Context manager for capturing logs.
