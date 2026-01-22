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
def HandleResumableUploadException(self, e, debug):
    if e.disposition == ResumableTransferDisposition.ABORT_CUR_PROCESS:
        if debug >= 1:
            self.logger.debug('Caught non-retryable ResumableUploadException (%s); aborting but retaining tracker file', e.message)
        raise
    elif e.disposition == ResumableTransferDisposition.ABORT:
        if debug >= 1:
            self.logger.debug('Caught non-retryable ResumableUploadException (%s); aborting and removing tracker file', e.message)
        raise
    elif e.disposition == ResumableTransferDisposition.START_OVER:
        raise
    elif debug >= 1:
        self.logger.debug('Caught ResumableUploadException (%s) - will retry', e.message)