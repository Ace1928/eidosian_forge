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
def __SendMediaBody(self, start, additional_headers=None):
    """Send the entire media stream in a single request."""
    self.EnsureInitialized()
    if self.total_size is None:
        raise exceptions.TransferInvalidError('Total size must be known for SendMediaBody')
    body_stream = stream_slice.StreamSlice(self.stream, self.total_size - start)
    request = http_wrapper.Request(url=self.url, http_method='PUT', body=body_stream)
    request.headers['Content-Type'] = self.mime_type
    if start == self.total_size:
        range_string = 'bytes */%s' % self.total_size
    else:
        range_string = 'bytes %s-%s/%s' % (start, self.total_size - 1, self.total_size)
    request.headers['Content-Range'] = range_string
    if additional_headers:
        request.headers.update(additional_headers)
    return self.__SendMediaRequest(request, self.total_size)