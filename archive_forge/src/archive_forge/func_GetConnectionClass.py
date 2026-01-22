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
def GetConnectionClass(self):
    """Returns a connection class that overrides getresponse."""

    class DownloadCallbackConnection(httplib2.HTTPSConnectionWithTimeout):
        """Connection class override for downloads."""
        outer_total_size = self.total_size
        outer_digesters = self.digesters
        outer_progress_callback = self.progress_callback
        outer_bytes_downloaded_container = self.bytes_downloaded_container
        processed_initial_bytes = False
        callback_processor = None

        def __init__(self, *args, **kwargs):
            kwargs['timeout'] = SSL_TIMEOUT_SEC
            httplib2.HTTPSConnectionWithTimeout.__init__(self, *args, **kwargs)

        def getresponse(self, buffering=False):
            """Wraps an HTTPResponse to perform callbacks and hashing.

        In this function, self is a DownloadCallbackConnection.

        Args:
          buffering: Unused. This function uses a local buffer.

        Returns:
          HTTPResponse object with wrapped read function.
        """
            orig_response = http_client.HTTPConnection.getresponse(self)
            if orig_response.status not in (http_client.OK, http_client.PARTIAL_CONTENT):
                return orig_response
            orig_read_func = orig_response.read

            def read(amt=None):
                """Overrides HTTPConnection.getresponse.read.

          This function only supports reads of TRANSFER_BUFFER_SIZE or smaller.

          Args:
            amt: Integer n where 0 < n <= TRANSFER_BUFFER_SIZE. This is a
                 keyword argument to match the read function it overrides,
                 but it is required.

          Returns:
            Data read from HTTPConnection.
          """
                if not amt or amt > TRANSFER_BUFFER_SIZE:
                    raise BadRequestException('Invalid HTTP read size %s during download, expected %s.' % (amt, TRANSFER_BUFFER_SIZE))
                else:
                    amt = amt or TRANSFER_BUFFER_SIZE
                if not self.processed_initial_bytes:
                    self.processed_initial_bytes = True
                    if self.outer_progress_callback:
                        self.callback_processor = ProgressCallbackWithTimeout(self.outer_total_size, self.outer_progress_callback)
                        self.callback_processor.Progress(self.outer_bytes_downloaded_container.bytes_transferred)
                data = orig_read_func(amt)
                read_length = len(data)
                if self.callback_processor:
                    self.callback_processor.Progress(read_length)
                if self.outer_digesters:
                    for alg in self.outer_digesters:
                        self.outer_digesters[alg].update(data)
                return data
            orig_response.read = read
            return orig_response
    return DownloadCallbackConnection