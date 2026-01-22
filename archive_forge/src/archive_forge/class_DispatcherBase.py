import inspect
import selectors
import socket
import threading
import time
from typing import Any, Callable, Optional, Union
from . import _logging
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
from ._url import parse_url
class DispatcherBase:
    """
    DispatcherBase
    """

    def __init__(self, app: Any, ping_timeout: Union[float, int, None]) -> None:
        self.app = app
        self.ping_timeout = ping_timeout

    def timeout(self, seconds: Union[float, int, None], callback: Callable) -> None:
        time.sleep(seconds)
        callback()

    def reconnect(self, seconds: int, reconnector: Callable) -> None:
        try:
            _logging.info(f'reconnect() - retrying in {seconds} seconds [{len(inspect.stack())} frames in stack]')
            time.sleep(seconds)
            reconnector(reconnecting=True)
        except KeyboardInterrupt as e:
            _logging.info(f'User exited {e}')
            raise e