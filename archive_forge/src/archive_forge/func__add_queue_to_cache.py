from __future__ import annotations
import string
from queue import Empty
from typing import Any, Dict, Set
import azure.core.exceptions
import azure.servicebus.exceptions
import isodate
from azure.servicebus import (ServiceBusClient, ServiceBusMessage,
from azure.servicebus.management import ServiceBusAdministrationClient
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _add_queue_to_cache(self, name: str, receiver: ServiceBusReceiver | None=None, sender: ServiceBusSender | None=None) -> SendReceive:
    if name in self._queue_cache:
        obj = self._queue_cache[name]
        obj.sender = obj.sender or sender
        obj.receiver = obj.receiver or receiver
    else:
        obj = SendReceive(receiver, sender)
        self._queue_cache[name] = obj
    return obj