from collections import OrderedDict
import logging
import threading
from typing import Dict, Optional, Type
import warnings
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
class _QuantityReservation:
    """A (partial) reservation of quantifiable resources."""

    def __init__(self, bytes_reserved: int, bytes_needed: int, has_slot: bool):
        self.bytes_reserved = bytes_reserved
        self.bytes_needed = bytes_needed
        self.has_slot = has_slot

    def __repr__(self):
        return f'{type(self).__name__}(bytes_reserved={self.bytes_reserved}, bytes_needed={self.bytes_needed}, has_slot={self.has_slot})'