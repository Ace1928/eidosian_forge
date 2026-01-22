from __future__ import annotations
import logging # isort:skip
import json
from typing import (
import bokeh.util.serialization as bkserial
from ..core.json_encoder import serialize_json
from ..core.serialization import Buffer, Serialized
from ..core.types import ID
from .exceptions import MessageError, ProtocolError
def assemble_buffer(self, buf_header: BufferHeader, buf_payload: bytes) -> None:
    """ Add a buffer header and payload that we read from the socket.

        This differs from add_buffer() because we're validating vs.
        the header's num_buffers, instead of filling in the header.

        Args:
            buf_header (``JSON``) : a buffer header
            buf_payload (``JSON`` or bytes) : a buffer payload

        Returns:
            None

        Raises:
            ProtocolError
        """
    num_buffers = self.header.get('num_buffers', 0)
    if num_buffers <= len(self._buffers):
        raise ProtocolError(f'too many buffers received expecting {num_buffers}')
    self._buffers.append(Buffer(buf_header['id'], buf_payload))