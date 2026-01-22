import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class StreamFailFast(StreamResult):
    """Call the supplied callback if an error is seen in a stream.

    An example callback::

       def do_something():
           pass
    """

    def __init__(self, on_error):
        self.on_error = on_error

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        if test_status in ('uxsuccess', 'fail'):
            self.on_error()