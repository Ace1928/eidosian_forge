from __future__ import annotations
import errno
import socket
from typing import TYPE_CHECKING
from amqp.exceptions import RecoverableConnectionError
from kombu.exceptions import ChannelError, ConnectionError
from kombu.message import Message
from kombu.utils.functional import dictfilter
from kombu.utils.objects import cached_property
from kombu.utils.time import maybe_s_to_ms
def _make_reader(self, connection, timeout=socket.timeout, error=socket.error, _unavail=(errno.EAGAIN, errno.EINTR)):
    drain_events = connection.drain_events

    def _read(loop):
        if not connection.connected:
            raise RecoverableConnectionError('Socket was disconnected')
        try:
            drain_events(timeout=0)
        except timeout:
            return
        except error as exc:
            if exc.errno in _unavail:
                return
            raise
        loop.call_soon(_read, loop)
    return _read