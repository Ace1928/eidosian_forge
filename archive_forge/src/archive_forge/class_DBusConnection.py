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
class DBusConnection(DBusConnectionBase):

    def __init__(self, sock: socket.socket, enable_fds=False):
        super().__init__(sock, enable_fds)
        self._router = Router(_Future)
        self._filters = MessageFilters()
        self.bus_proxy = Proxy(message_bus, self)
        hello_reply = self.bus_proxy.Hello()
        self.unique_name = hello_reply[0]

    @property
    def router(self):
        warn('conn.router is deprecated, see the docs for APIs to use instead.', stacklevel=2)
        return self._router

    def send(self, message: Message, serial=None):
        """Serialise and send a :class:`~.Message` object"""
        data, fds = self._serialise(message, serial)
        if fds:
            self._send_with_fds(data, fds)
        else:
            self.sock.sendall(data)
    send_message = send

    def receive(self, *, timeout=None) -> Message:
        """Return the next available message from the connection

        If the data is ready, this will return immediately, even if timeout<=0.
        Otherwise, it will wait for up to timeout seconds, or indefinitely if
        timeout is None. If no message comes in time, it raises TimeoutError.
        """
        return self._receive(timeout_to_deadline(timeout))

    def recv_messages(self, *, timeout=None):
        """Receive one message and apply filters

        See :meth:`filter`. Returns nothing.
        """
        msg = self.receive(timeout=timeout)
        self._router.incoming(msg)
        for filter in self._filters.matches(msg):
            filter.queue.append(msg)

    def send_and_get_reply(self, message, *, timeout=None, unwrap=None):
        """Send a message, wait for the reply and return it

        Filters are applied to other messages received before the reply -
        see :meth:`add_filter`.
        """
        check_replyable(message)
        deadline = timeout_to_deadline(timeout)
        if unwrap is None:
            unwrap = False
        else:
            warn('Passing unwrap= to .send_and_get_reply() is deprecated and will break in a future version of Jeepney.', stacklevel=2)
        serial = next(self.outgoing_serial)
        self.send_message(message, serial=serial)
        while True:
            msg_in = self.receive(timeout=deadline_to_timeout(deadline))
            reply_to = msg_in.header.fields.get(HeaderFields.reply_serial, -1)
            if reply_to == serial:
                if unwrap:
                    return unwrap_msg(msg_in)
                return msg_in
            self._router.incoming(msg_in)
            for filter in self._filters.matches(msg_in):
                filter.queue.append(msg_in)

    def filter(self, rule, *, queue: Optional[deque]=None, bufsize=1):
        """Create a filter for incoming messages

        Usage::

            with conn.filter(rule) as matches:
                # matches is a deque containing matched messages
                matching_msg = conn.recv_until_filtered(matches)

        :param jeepney.MatchRule rule: Catch messages matching this rule
        :param collections.deque queue: Matched messages will be added to this
        :param int bufsize: If no deque is passed in, create one with this size
        """
        if queue is None:
            queue = deque(maxlen=bufsize)
        return FilterHandle(self._filters, rule, queue)

    def recv_until_filtered(self, queue, *, timeout=None) -> Message:
        """Process incoming messages until one is filtered into queue

        Pops the message from queue and returns it, or raises TimeoutError if
        the optional timeout expires. Without a timeout, this is equivalent to::

            while len(queue) == 0:
                conn.recv_messages()
            return queue.popleft()

        In the other I/O modules, there is no need for this, because messages
        are placed in queues by a separate task.

        :param collections.deque queue: A deque connected by :meth:`filter`
        :param float timeout: Maximum time to wait in seconds
        """
        deadline = timeout_to_deadline(timeout)
        while len(queue) == 0:
            self.recv_messages(timeout=deadline_to_timeout(deadline))
        return queue.popleft()