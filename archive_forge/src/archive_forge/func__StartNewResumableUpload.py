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
def _StartNewResumableUpload(self, key, headers=None):
    """Starts a new resumable upload.

    Args:
      key: Boto Key representing the object to upload.
      headers: Headers to use in the upload requests.

    Raises:
      ResumableUploadException if any errors occur.
    """
    conn = key.bucket.connection
    if conn.debug >= 1:
        self.logger.debug('Starting new resumable upload.')
    self.service_has_bytes = 0
    post_headers = {}
    for k in headers:
        if k.lower() == 'content-length':
            raise ResumableUploadException('Attempt to specify Content-Length header (disallowed)', ResumableTransferDisposition.ABORT)
        post_headers[k] = headers[k]
    post_headers[conn.provider.resumable_upload_header] = 'start'
    resp = conn.make_request('POST', key.bucket.name, key.name, post_headers)
    body = resp.read()
    if resp.status in [429, 500, 503]:
        raise ResumableUploadException('Got status %d from attempt to start resumable upload. Will wait/retry' % resp.status, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
    elif resp.status != 200 and resp.status != 201:
        raise ResumableUploadException('Got status %d from attempt to start resumable upload. Aborting' % resp.status, ResumableTransferDisposition.ABORT)
    upload_url = resp.getheader('Location')
    if not upload_url:
        raise ResumableUploadException('No resumable upload URL found in resumable initiation POST response (%s)' % body, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
    self._SetUploadUrl(upload_url)
    self.tracker_callback(upload_url)