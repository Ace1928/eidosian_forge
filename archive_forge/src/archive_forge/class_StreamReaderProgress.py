import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
class StreamReaderProgress(object):
    """Wrapper for a read only stream that adds progress callbacks."""

    def __init__(self, stream, callback=None):
        self._stream = stream
        self._callback = callback

    def read(self, *args, **kwargs):
        value = self._stream.read(*args, **kwargs)
        if self._callback is not None:
            self._callback(len(value))
        return value