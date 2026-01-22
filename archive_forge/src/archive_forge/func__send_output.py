from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import copy
import logging
import re
import socket
import types
import six
from six.moves import http_client
from six.moves import urllib
from six.moves import cStringIO
from apitools.base.py import exceptions as apitools_exceptions
from gslib.cloud_api import BadRequestException
from gslib.lazy_wrapper import LazyWrapper
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
from gslib.utils import text_util
import httplib2
from httplib2 import parse_uri
def _send_output(self, message_body=None, encode_chunked=False):
    """Send the currently buffered request and clear the buffer.

        Appends an extra \\r\\n to the buffer.

        Args:
          message_body: if specified, this is appended to the request.
        """
    self._buffer.extend((b'', b''))
    if six.PY2:
        items = self._buffer
    else:
        items = []
        for item in self._buffer:
            if isinstance(item, bytes):
                items.append(item)
            else:
                items.append(item.encode(UTF8))
    msg = b'\r\n'.join(items)
    num_metadata_bytes = len(msg)
    if outer_debug == DEBUGLEVEL_DUMP_REQUESTS and outer_logger:
        outer_logger.debug('send: %s' % msg)
    del self._buffer[:]
    if isinstance(message_body, str):
        msg += message_body
        message_body = None
    self.send(msg, num_metadata_bytes=num_metadata_bytes)
    if message_body is not None:
        self.send(message_body)