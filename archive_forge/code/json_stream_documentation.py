from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import json
import six
Python 2 compatibility with py3's streaming behavior.

    If file_obj is an HTTPResponse, iterating over lines blocks until a buffer
    is full.

    Args:
      file_obj: A file-like object, including HTTPResponse.

    Yields:
      Lines, like iter(file_obj) but without buffering stalls.
    