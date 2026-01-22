import ssl
import sys
from types import TracebackType
from typing import Iterable, Iterator, Iterable, List, Optional, Type
from .._backends.sync import SyncBackend
from .._backends.base import SOCKET_OPTION, NetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Request, Response
from .._synchronization import Event, ShieldCancellation, ThreadLock
from .connection import HTTPConnection
from .interfaces import ConnectionInterface, RequestInterface
class PoolRequest:

    def __init__(self, request: Request) -> None:
        self.request = request
        self.connection: Optional[ConnectionInterface] = None
        self._connection_acquired = Event()

    def assign_to_connection(self, connection: Optional[ConnectionInterface]) -> None:
        self.connection = connection
        self._connection_acquired.set()

    def clear_connection(self) -> None:
        self.connection = None
        self._connection_acquired = Event()

    def wait_for_connection(self, timeout: Optional[float]=None) -> ConnectionInterface:
        if self.connection is None:
            self._connection_acquired.wait(timeout=timeout)
        assert self.connection is not None
        return self.connection

    def is_queued(self) -> bool:
        return self.connection is None