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
class SendReceive:
    """Container for Sender and Receiver."""

    def __init__(self, receiver: ServiceBusReceiver | None=None, sender: ServiceBusSender | None=None):
        self.receiver: ServiceBusReceiver = receiver
        self.sender: ServiceBusSender = sender

    def close(self) -> None:
        if self.receiver:
            self.receiver.close()
            self.receiver = None
        if self.sender:
            self.sender.close()
            self.sender = None