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
def __SetDefaultUploadStrategy(self, upload_config, http_request):
    """Determine and set the default upload strategy for this upload.

        We generally prefer simple or multipart, unless we're forced to
        use resumable. This happens when any of (1) the upload is too
        large, (2) the simple endpoint doesn't support multipart requests
        and we have metadata, or (3) there is no simple upload endpoint.

        Args:
          upload_config: Configuration for the upload endpoint.
          http_request: The associated http request.

        Returns:
          None.
        """
    if upload_config.resumable_path is None:
        self.strategy = SIMPLE_UPLOAD
    if self.strategy is not None:
        return
    strategy = SIMPLE_UPLOAD
    if self.total_size is not None and self.total_size > _RESUMABLE_UPLOAD_THRESHOLD:
        strategy = RESUMABLE_UPLOAD
    if http_request.body and (not upload_config.simple_multipart):
        strategy = RESUMABLE_UPLOAD
    if not upload_config.simple_path:
        strategy = RESUMABLE_UPLOAD
    self.strategy = strategy