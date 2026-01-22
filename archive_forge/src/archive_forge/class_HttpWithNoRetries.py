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
class HttpWithNoRetries(httplib2.Http):
    """httplib2.Http variant that does not retry.

  httplib2 automatically retries requests according to httplib2.RETRIES, but
  in certain cases httplib2 ignores the RETRIES value and forces a retry.
  Because httplib2 does not handle the case where the underlying request body
  is a stream, a retry may cause a non-idempotent write as the stream is
  partially consumed and not reset before the retry occurs.

  Here we override _conn_request to disable retries unequivocally, so that
  uploads may be retried at higher layers that properly handle stream request
  bodies.
  """

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
        except (socket.error, http_client.HTTPException):
            conn.close()
            raise
        else:
            content = ''
            if method == 'HEAD':
                conn.close()
            else:
                content = response.read()
            response = httplib2.Response(response)
            if method != 'HEAD':
                content = httplib2._decompressContent(response, content)
        return (response, content)