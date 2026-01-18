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
def test_timestamps(self):
    tests = []
    result = StreamToDict(tests.append)
    result.startTestRun()
    result.status(test_id='foo', test_status='inprogress', timestamp='A')
    result.status(test_id='foo', test_status='success', timestamp='B')
    result.status(test_id='bar', test_status='inprogress', timestamp='C')
    result.stopTestRun()
    self.assertThat(tests, HasLength(2))
    self.assertEqual(['A', 'B'], tests[0]['timestamps'])
    self.assertEqual(['C', None], tests[1]['timestamps'])