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
def _get_asb_receiver(self, queue: str, recv_mode: ServiceBusReceiveMode=ServiceBusReceiveMode.PEEK_LOCK, queue_cache_key: str | None=None) -> SendReceive:
    cache_key = queue_cache_key or queue
    queue_obj = self._queue_cache.get(cache_key, None)
    if queue_obj is None or queue_obj.receiver is None:
        receiver = self.queue_service.get_queue_receiver(queue_name=queue, receive_mode=recv_mode, keep_alive=self.uamqp_keep_alive_interval)
        queue_obj = self._add_queue_to_cache(cache_key, receiver=receiver)
    return queue_obj