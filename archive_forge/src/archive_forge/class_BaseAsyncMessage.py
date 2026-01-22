from __future__ import annotations
import base64
from kombu.message import Message
from kombu.utils.encoding import str_to_bytes
class BaseAsyncMessage(Message):
    """Base class for messages received on async client."""