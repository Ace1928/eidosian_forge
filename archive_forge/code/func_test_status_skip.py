import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_status_skip(self):
    result = StreamSummary()
    result.startTestRun()
    result.status(file_name='reason', file_bytes=_b('Missing dependency'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
    result.status('foo.bar', 'skip')
    self.assertThat(result.skipped, HasLength(1))
    self.assertEqual('foo.bar', result.skipped[0][0].id())
    self.assertEqual('Missing dependency', result.skipped[0][1])