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
def test_test_status(self):
    result = self._make_result()
    result.startTestRun()
    self.addCleanup(result.stopTestRun)
    now = datetime.datetime.now(utc)
    args = [['foo', s] for s in ['exists', 'inprogress', 'xfail', 'uxsuccess', 'success', 'fail', 'skip']]
    inputs = list(dict(runnable=False, test_tags={'quux'}, route_code='1234', timestamp=now).items())
    param_dicts = self._power_set(inputs)
    for kwargs in param_dicts:
        for arg in args:
            result.status(test_id=arg[0], test_status=arg[1], **kwargs)