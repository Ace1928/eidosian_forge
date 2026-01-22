import array
from contextlib import contextmanager
import errno
from itertools import count
import logging
from typing import Optional
from outcome import Value, Error
import trio
from trio.abc import Channel
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.low_level import Parser, MessageType, Message
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.bus_messages import message_bus
from .common import (
class TrioFilterHandle(FilterHandle):

    def __init__(self, filters: MessageFilters, rule, send_chn, recv_chn):
        super().__init__(filters, rule, recv_chn)
        self.send_channel = send_chn

    @property
    def receive_channel(self):
        return self.queue

    async def aclose(self):
        self.close()
        await self.send_channel.aclose()

    async def __aenter__(self):
        return self.queue

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()