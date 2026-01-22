import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def _send_connection_init(self, request: Request) -> None:
    """
        The HTTP/2 connection requires some initial setup before we can start
        using individual request/response streams on it.
        """
    self._h2_state.local_settings = h2.settings.Settings(client=True, initial_values={h2.settings.SettingCodes.ENABLE_PUSH: 0, h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 100, h2.settings.SettingCodes.MAX_HEADER_LIST_SIZE: 65536})
    del self._h2_state.local_settings[h2.settings.SettingCodes.ENABLE_CONNECT_PROTOCOL]
    self._h2_state.initiate_connection()
    self._h2_state.increment_flow_control_window(2 ** 24)
    self._write_outgoing_data(request)