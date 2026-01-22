import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def _handleTime(self, offset, line):
    try:
        event_time = iso8601.parse_date(line[offset:-1].decode())
    except TypeError:
        raise TypeError(_u('Failed to parse %r, got %r') % (line, sys.exc_info()[1]))
    self.client.time(event_time)