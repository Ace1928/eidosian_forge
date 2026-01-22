from __future__ import annotations
import base64
from kombu.message import Message
from kombu.utils.encoding import str_to_bytes
class AsyncMessage(BaseAsyncMessage):
    """Serialized message."""

    def encode(self, value):
        """Encode/decode the value using Base64 encoding."""
        return base64.b64encode(str_to_bytes(value)).decode()

    def __getitem__(self, item):
        """Support Boto3-style access on a message."""
        if item == 'ReceiptHandle':
            return self.receipt_handle
        elif item == 'Body':
            return self.get_body()
        elif item == 'queue':
            return self.queue
        else:
            raise KeyError(item)