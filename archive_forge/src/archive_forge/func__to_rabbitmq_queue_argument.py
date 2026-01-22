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
def _to_rabbitmq_queue_argument(key, value):
    opt, typ = RABBITMQ_QUEUE_ARGUMENTS[key]
    return (opt, typ(value) if value is not None else value)