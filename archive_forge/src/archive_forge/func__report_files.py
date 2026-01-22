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
def _report_files(self, result):
    result.status(file_name='some log.txt', file_bytes=_b('1234 log message'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
    result.status(file_name='traceback', file_bytes=_b('Traceback (most recent call last):\n  File "testtools/tests/test_testresult.py", line 607, in test_stopTestRun\n      AllMatch(Equals([(\'startTestRun\',), (\'stopTestRun\',)])))\ntesttools.matchers._impl.MismatchError: Differences: [\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n[(\'startTestRun\',), (\'stopTestRun\',)] != []\n]\n'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')