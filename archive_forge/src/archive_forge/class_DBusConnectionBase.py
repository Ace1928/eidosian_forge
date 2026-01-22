import array
from collections import deque
from errno import ECONNRESET
import functools
from itertools import count
import os
from selectors import DefaultSelector, EVENT_READ
import socket
import time
from typing import Optional
from warnings import warn
from jeepney import Parser, Message, MessageType, HeaderFields
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.routing import Router
from jeepney.bus_messages import message_bus
from .common import MessageFilters, FilterHandle, check_replyable
class DBusConnectionBase:
    """Connection machinery shared by this module and threading"""

    def __init__(self, sock: socket.socket, enable_fds=False):
        self.sock = sock
        self.enable_fds = enable_fds
        self.parser = Parser()
        self.outgoing_serial = count(start=1)
        self.selector = DefaultSelector()
        self.select_key = self.selector.register(sock, EVENT_READ)
        self.unique_name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _serialise(self, message: Message, serial) -> (bytes, Optional[array.array]):
        if serial is None:
            serial = next(self.outgoing_serial)
        fds = array.array('i') if self.enable_fds else None
        data = message.serialise(serial=serial, fds=fds)
        return (data, fds)

    def _send_with_fds(self, data, fds):
        bytes_sent = self.sock.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
        if bytes_sent < len(data):
            self.sock.sendall(data[bytes_sent:])

    def _receive(self, deadline):
        while True:
            msg = self.parser.get_next_message()
            if msg is not None:
                return msg
            b, fds = self._read_some_data(timeout=deadline_to_timeout(deadline))
            self.parser.add_data(b, fds=fds)

    def _read_some_data(self, timeout=None):
        for key, ev in self.selector.select(timeout):
            if key == self.select_key:
                if self.enable_fds:
                    return self._read_with_fds()
                else:
                    return (unwrap_read(self.sock.recv(4096)), [])
        raise TimeoutError

    def _read_with_fds(self):
        nbytes = self.parser.bytes_desired()
        data, ancdata, flags, _ = self.sock.recvmsg(nbytes, fds_buf_size())
        if flags & getattr(socket, 'MSG_CTRUNC', 0):
            self.close()
            raise RuntimeError('Unable to receive all file descriptors')
        return (unwrap_read(data), FileDescriptor.from_ancdata(ancdata))

    def close(self):
        """Close the connection"""
        self.selector.close()
        self.sock.close()