from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
class ProtocolBufferModel(BaseModel):
    """Model class for protocol buffers.

  Serializes and de-serializes the binary protocol buffer sent in the HTTP
  request and response bodies.
  """
    accept = 'application/x-protobuf'
    content_type = 'application/x-protobuf'
    alt_param = 'proto'

    def __init__(self, protocol_buffer):
        """Constructs a ProtocolBufferModel.

    The serialized protocol buffer returned in an HTTP response will be
    de-serialized using the given protocol buffer class.

    Args:
      protocol_buffer: The protocol buffer class used to de-serialize a
      response from the API.
    """
        self._protocol_buffer = protocol_buffer

    def serialize(self, body_value):
        return body_value.SerializeToString()

    def deserialize(self, content):
        return self._protocol_buffer.FromString(content)

    @property
    def no_content_response(self):
        return self._protocol_buffer()