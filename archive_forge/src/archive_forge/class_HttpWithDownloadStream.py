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
class HttpWithDownloadStream(httplib2.Http):
    """httplib2.Http variant that only pushes bytes through a stream.

  httplib2 handles media by storing entire chunks of responses in memory, which
  is undesirable particularly when multiple instances are used during
  multi-threaded/multi-process copy. This class copies and then overrides some
  httplib2 functions to use a streaming copy approach that uses small memory
  buffers.

  Also disables httplib2 retries (for reasons stated in the HttpWithNoRetries
  class doc).
  """

    def __init__(self, *args, **kwds):
        self._stream = None
        self._logger = logging.getLogger()
        super(HttpWithDownloadStream, self).__init__(*args, **kwds)

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def _conn_request(self, conn, request_uri, method, body, headers):
        try:
            if hasattr(conn, 'sock') and conn.sock is None:
                conn.connect()
            conn.request(method, request_uri, body, headers)
        except socket.timeout:
            raise
        except socket.gaierror:
            conn.close()
            raise httplib2.ServerNotFoundError('Unable to find the server at %s' % conn.host)
        except httplib2.ssl.SSLError:
            conn.close()
            raise
        except socket.error as e:
            err = 0
            if hasattr(e, 'args'):
                err = getattr(e, 'args')[0]
            else:
                err = e.errno
            if err == httplib2.errno.ECONNREFUSED:
                raise
        except http_client.HTTPException:
            conn.close()
            raise
        try:
            response = conn.getresponse()
        except (socket.error, http_client.HTTPException) as e:
            conn.close()
            raise
        else:
            content = ''
            if method == 'HEAD':
                conn.close()
                response = httplib2.Response(response)
            elif method == 'GET' and response.status in (http_client.OK, http_client.PARTIAL_CONTENT):
                content_length = None
                if hasattr(response, 'msg'):
                    content_length = response.getheader('content-length')
                http_stream = response
                bytes_read = 0
                while True:
                    new_data = http_stream.read(TRANSFER_BUFFER_SIZE)
                    if new_data:
                        if self.stream is None:
                            raise apitools_exceptions.InvalidUserInputError('Cannot exercise HttpWithDownloadStream with no stream')
                        text_util.write_to_fd(self.stream, new_data)
                        bytes_read += len(new_data)
                    else:
                        break
                if content_length is not None and long(bytes_read) != long(content_length):
                    self._logger.log(logging.DEBUG, 'Only got %s bytes out of content-length %s for request URI %s. Resetting content-length to match bytes read.', bytes_read, content_length, request_uri)
                    del response.msg['content-length']
                    response.msg['content-length'] = str(bytes_read)
                response = httplib2.Response(response)
            else:
                content = response.read()
                response = httplib2.Response(response)
                content = httplib2._decompressContent(response, content)
        return (response, content)