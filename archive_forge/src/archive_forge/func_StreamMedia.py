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
def StreamMedia(self, callback=None, finish_callback=None, additional_headers=None):
    """Send this resumable upload in a single request.

        Args:
          callback: Progress callback function with inputs
              (http_wrapper.Response, transfer.Upload)
          finish_callback: Final callback function with inputs
              (http_wrapper.Response, transfer.Upload)
          additional_headers: Dict of headers to include with the upload
              http_wrapper.Request.

        Returns:
          http_wrapper.Response of final response.
        """
    return self.__StreamMedia(callback=callback, finish_callback=finish_callback, additional_headers=additional_headers, use_chunks=False)