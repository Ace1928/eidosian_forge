import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _remove_tag(self, consumer_tag):
    self.callbacks.pop(consumer_tag, None)
    return self.cancel_callbacks.pop(consumer_tag, None)