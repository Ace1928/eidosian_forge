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
class TestNonAsciiResultsWithUnittest(TestNonAsciiResults):
    """Test that running under unittest produces clean ascii strings"""

    def _run(self, stream, test):
        from unittest import TextTestRunner as _Runner
        return _Runner(stream).run(test)

    def _as_output(self, text):
        return text