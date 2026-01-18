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
def test_stopTestRun_not_successful_unexpected_success(self):
    test = make_unexpectedly_successful_test()
    self.result.startTestRun()
    test.run(self.result)
    self.result.stopTestRun()
    self.assertThat(self.getvalue(), DocTestMatches('...\nFAILED (failures=1)\n', doctest.ELLIPSIS))