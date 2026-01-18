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
def test_now_datetime_now(self):
    result = self.makeResult()
    olddatetime = testresult.real.datetime

    def restore():
        testresult.real.datetime = olddatetime
    self.addCleanup(restore)

    class Module:
        pass
    now = datetime.datetime.now(utc)
    stubdatetime = Module()
    stubdatetime.datetime = Module()
    stubdatetime.datetime.now = lambda tz: now
    testresult.real.datetime = stubdatetime
    self.assertEqual(now, result._now())
    then = now + datetime.timedelta(0, 1)
    result.time(then)
    self.assertNotEqual(now, result._now())
    self.assertEqual(then, result._now())
    result.time(None)
    self.assertEqual(now, result._now())