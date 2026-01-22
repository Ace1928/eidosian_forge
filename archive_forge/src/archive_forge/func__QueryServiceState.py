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
def _QueryServiceState(self, conn, file_length):
    """Queries service to find out state of given upload.

    Note that this method really just makes special case use of the
    fact that the upload service always returns the current start/end
    state whenever a PUT doesn't complete.

    Args:
      conn: HTTPConnection to use for the query.
      file_length: Total length of the file.

    Returns:
      HTTP response from sending request.

    Raises:
      ResumableUploadException if problem querying service.
    """
    put_headers = {'Content-Range': self._BuildContentRangeHeader('*', file_length), 'Content-Length': '0'}
    return AWSAuthConnection.make_request(conn, 'PUT', path=self.upload_url_path, auth_path=self.upload_url_path, headers=put_headers, host=self.upload_url_host)