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
def __SetTotal(self, info):
    """Sets the total size based off info if possible otherwise 0."""
    if 'content-range' in info:
        _, _, total = info['content-range'].rpartition('/')
        if total != '*':
            self.__total_size = int(total)
    if self.total_size is None:
        self.__total_size = 0