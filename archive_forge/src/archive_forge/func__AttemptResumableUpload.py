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
def _AttemptResumableUpload(self, key, fp, file_length, headers, cb, num_cb):
    """Attempts a resumable upload.

    Args:
      key: Boto key representing object to upload.
      fp: File pointer containing upload bytes.
      file_length: Total length of the upload.
      headers: Headers to be used in upload requests.
      cb: Progress callback function that takes (progress, total_size).
      num_cb: Granularity of the callback (maximum number of times the
              callback will be called during the file transfer). If negative,
              perform callback with each buffer read.

    Returns:
      (etag, generation, metageneration) from service upon success.

    Raises:
      ResumableUploadException if any problems occur.
    """
    service_start, service_end = self.SERVICE_HAS_NOTHING
    conn = key.bucket.connection
    if self.upload_url:
        try:
            service_start, service_end = self._QueryServicePos(conn, file_length)
            self.service_has_bytes = service_start
            if conn.debug >= 1:
                self.logger.debug('Resuming transfer.')
        except ResumableUploadException as e:
            if conn.debug >= 1:
                self.logger.debug('Unable to resume transfer (%s).', e.message)
            self._StartNewResumableUpload(key, headers)
    else:
        self._StartNewResumableUpload(key, headers)
    if self.upload_start_point is None:
        self.upload_start_point = service_end
    total_bytes_uploaded = service_end + 1
    if total_bytes_uploaded < file_length:
        fp.seek(total_bytes_uploaded)
    conn = key.bucket.connection
    http_conn = conn.new_http_connection(self.upload_url_host, conn.port, conn.is_secure)
    http_conn.set_debuglevel(conn.debug)
    try:
        return self._UploadFileBytes(conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers)
    except (ResumableUploadException, socket.error):
        resp = self._QueryServiceState(conn, file_length)
        if resp.status == 400:
            raise ResumableUploadException('Got 400 response from service state query after failed resumable upload attempt. This can happen for various reasons, including specifying an invalid request (e.g., an invalid canned ACL) or if the file size changed between upload attempts', ResumableTransferDisposition.ABORT)
        else:
            raise
    finally:
        http_conn.close()