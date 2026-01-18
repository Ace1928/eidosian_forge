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
Overrides HTTPConnection.getresponse.read.

          This function only supports reads of TRANSFER_BUFFER_SIZE or smaller.

          Args:
            amt: Integer n where 0 < n <= TRANSFER_BUFFER_SIZE. This is a
                 keyword argument to match the read function it overrides,
                 but it is required.

          Returns:
            Data read from HTTPConnection.
          