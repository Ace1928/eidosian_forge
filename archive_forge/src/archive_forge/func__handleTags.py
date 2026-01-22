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
def _handleTags(self, offset, line):
    """Process a tags command."""
    tags = line[offset:].decode('utf8').split()
    new_tags, gone_tags = tags_to_new_gone(tags)
    self.client.tags(new_tags, gone_tags)