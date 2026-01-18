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
def to_rabbitmq_queue_arguments(arguments, **options):
    """Convert queue arguments to RabbitMQ queue arguments.

    This is the implementation for Channel.prepare_queue_arguments
    for AMQP-based transports.  It's used by both the pyamqp and librabbitmq
    transports.

    Arguments:
        arguments (Mapping):
            User-supplied arguments (``Queue.queue_arguments``).

    Keyword Arguments:
        expires (float): Queue expiry time in seconds.
            This will be converted to ``x-expires`` in int milliseconds.
        message_ttl (float): Message TTL in seconds.
            This will be converted to ``x-message-ttl`` in int milliseconds.
        max_length (int): Max queue length (in number of messages).
            This will be converted to ``x-max-length`` int.
        max_length_bytes (int): Max queue size in bytes.
            This will be converted to ``x-max-length-bytes`` int.
        max_priority (int): Max priority steps for queue.
            This will be converted to ``x-max-priority`` int.

    Returns
    -------
        Dict: RabbitMQ compatible queue arguments.
    """
    prepared = dictfilter(dict((_to_rabbitmq_queue_argument(key, value) for key, value in options.items())))
    return dict(arguments, **prepared) if prepared else arguments