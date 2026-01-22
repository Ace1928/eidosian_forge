from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
def _Initialize(self, http, url):
    """Initialize this download by setting self.http and self.url.

        We want the user to be able to override self.http by having set
        the value in the constructor; in that case, we ignore the provided
        http.

        Args:
          http: An httplib2.Http instance or None.
          url: The url for this transfer.

        Returns:
          None. Initializes self.
        """
    self.EnsureUninitialized()
    if self.http is None:
        self.__http = http or http_wrapper.GetHttp()
    self.__url = url