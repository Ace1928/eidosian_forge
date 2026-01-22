from __future__ import annotations
import os
import select
import socket
import ssl
import sys
import uuid
from gettext import gettext as _
from queue import Empty
from time import monotonic
import amqp.protocol
from kombu.log import get_logger
from kombu.transport import base, virtual
from kombu.transport.virtual import Base64, Message
def _qpid_async_exception_notify_handler(self, obj_with_exception, exc):
    if self.use_async_interface:
        os.write(self._w, 'e')