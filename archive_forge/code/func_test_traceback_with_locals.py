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
def test_traceback_with_locals(self):
    result = self.makeResult()
    result.tb_locals = True
    test = make_erroring_test()
    test.run(result)
    self.assertThat(result.errors[0][1], DocTestMatches('Traceback (most recent call last):\n  File "...testtools...runtest.py", line ..., in _run_user\n    return fn(*args, **kwargs)\n...    args = ...\n    fn = ...\n    kwargs = ...\n    self = ...\n  File "...testtools...testcase.py", line ..., in _run_test_method\n    return self._get_test_method()()\n...    result = ...\n    self = ...\n  File "...testtools...tests...test_testresult.py", line ..., in error\n    1/0\n...    a = 1\n    self = ...\nZeroDivisionError: ...\n', doctest.ELLIPSIS | doctest.REPORT_UDIFF))