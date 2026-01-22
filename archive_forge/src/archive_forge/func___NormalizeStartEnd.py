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
def __NormalizeStartEnd(self, start, end=None):
    """Normalizes start and end values based on total size."""
    if end is not None:
        if start < 0:
            raise exceptions.TransferInvalidError('Cannot have end index with negative start index ' + '[start=%d, end=%d]' % (start, end))
        elif start >= self.total_size:
            raise exceptions.TransferInvalidError('Cannot have start index greater than total size ' + '[start=%d, total_size=%d]' % (start, self.total_size))
        end = min(end, self.total_size - 1)
        if end < start:
            raise exceptions.TransferInvalidError('Range requested with end[%s] < start[%s]' % (end, start))
        return (start, end)
    else:
        if start < 0:
            start = max(0, start + self.total_size)
        return (start, self.total_size - 1)