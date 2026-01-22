from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import random
import re
import socket
import time
import six
from six.moves import urllib
from six.moves import http_client
from boto import config
from boto import UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from gslib.exception import InvalidUrlError
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.constants import UTF8
def _UploadFileBytes(self, conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers):
    """Attempts to upload file bytes.

    Makes a single attempt using an existing resumable upload connection.

    Args:
      conn: HTTPConnection from the boto Key.
      http_conn: Separate HTTPConnection for the transfer.
      fp: File pointer containing bytes to upload.
      file_length: Total length of the file.
      total_bytes_uploaded: The total number of bytes uploaded.
      cb: Progress callback function that takes (progress, total_size).
      num_cb: Granularity of the callback (maximum number of times the
              callback will be called during the file transfer). If negative,
              perform callback with each buffer read.
      headers: Headers to be used in the upload requests.

    Returns:
      (etag, generation, metageneration) from service upon success.

    Raises:
      ResumableUploadException if any problems occur.
    """
    buf = fp.read(self.BUFFER_SIZE)
    if cb:
        if num_cb > 2:
            cb_count = file_length / self.BUFFER_SIZE / (num_cb - 2)
        elif num_cb < 0:
            cb_count = -1
        else:
            cb_count = 0
        i = 0
        cb(total_bytes_uploaded, file_length)
    put_headers = headers.copy() if headers else {}
    if file_length:
        if total_bytes_uploaded == file_length:
            range_header = self._BuildContentRangeHeader('*', file_length)
        else:
            range_header = self._BuildContentRangeHeader('%d-%d' % (total_bytes_uploaded, file_length - 1), file_length)
        put_headers['Content-Range'] = range_header
    put_headers['Content-Length'] = str(file_length - total_bytes_uploaded)
    http_request = AWSAuthConnection.build_base_http_request(conn, 'PUT', path=self.upload_url_path, auth_path=None, headers=put_headers, host=self.upload_url_host)
    http_conn.putrequest('PUT', http_request.path)
    for k in put_headers:
        http_conn.putheader(k, put_headers[k])
    http_conn.endheaders()
    http_conn.set_debuglevel(0)
    while buf:
        if six.PY2:
            http_conn.send(buf)
            total_bytes_uploaded += len(buf)
        elif isinstance(buf, bytes):
            http_conn.send(buf)
            total_bytes_uploaded += len(buf)
        else:
            buf_bytes = buf.encode(UTF8)
            http_conn.send(buf_bytes)
            total_bytes_uploaded += len(buf_bytes)
        if cb:
            i += 1
            if i == cb_count or cb_count == -1:
                cb(total_bytes_uploaded, file_length)
                i = 0
        buf = fp.read(self.BUFFER_SIZE)
    http_conn.set_debuglevel(conn.debug)
    if cb:
        cb(total_bytes_uploaded, file_length)
    if total_bytes_uploaded != file_length:
        raise ResumableUploadException('File changed during upload: EOF at %d bytes of %d byte file.' % (total_bytes_uploaded, file_length), ResumableTransferDisposition.ABORT)
    resp = http_conn.getresponse()
    if resp.status == 200:
        return (resp.getheader('etag'), resp.getheader('x-goog-generation'), resp.getheader('x-goog-metageneration'))
    elif resp.status in [408, 429, 500, 503]:
        disposition = ResumableTransferDisposition.WAIT_BEFORE_RETRY
    else:
        disposition = ResumableTransferDisposition.ABORT
    raise ResumableUploadException('Got response code %d while attempting upload (%s)' % (resp.status, resp.reason), disposition)