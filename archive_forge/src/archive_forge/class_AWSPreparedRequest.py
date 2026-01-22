import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
class AWSPreparedRequest:
    """A data class representing a finalized request to be sent over the wire.

    Requests at this stage should be treated as final, and the properties of
    the request should not be modified.

    :ivar method: The HTTP Method
    :ivar url: The full url
    :ivar headers: The HTTP headers to send.
    :ivar body: The HTTP body.
    :ivar stream_output: If the response for this request should be streamed.
    """

    def __init__(self, method, url, headers, body, stream_output):
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body
        self.stream_output = stream_output

    def __repr__(self):
        fmt = '<AWSPreparedRequest stream_output=%s, method=%s, url=%s, headers=%s>'
        return fmt % (self.stream_output, self.method, self.url, self.headers)

    def reset_stream(self):
        """Resets the streaming body to it's initial position.

        If the request contains a streaming body (a streamable file-like object)
        seek to the object's initial position to ensure the entire contents of
        the object is sent. This is a no-op for static bytes-like body types.
        """
        non_seekable_types = (bytes, str, bytearray)
        if self.body is None or isinstance(self.body, non_seekable_types):
            return
        try:
            logger.debug('Rewinding stream: %s', self.body)
            self.body.seek(0)
        except Exception as e:
            logger.debug('Unable to rewind stream: %s', e)
            raise UnseekableStreamError(stream_object=self.body)