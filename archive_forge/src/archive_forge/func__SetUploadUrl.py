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
def _SetUploadUrl(self, url):
    """Saves URL and resets upload state.

    Called when we start a new resumable upload or get a new tracker
    URL for the upload.

    Args:
      url: URL string for the upload.

    Raises InvalidUrlError if URL is syntactically invalid.
    """
    parse_result = urllib.parse.urlparse(url)
    if parse_result.scheme.lower() not in ['http', 'https'] or not parse_result.netloc:
        raise InvalidUrlError('Invalid upload URL (%s)' % url)
    self.upload_url = url
    self.upload_url_host = config.get('Credentials', 'gs_host', None) or parse_result.netloc
    self.upload_url_path = '%s?%s' % (parse_result.path, parse_result.query)
    self.service_has_bytes = 0