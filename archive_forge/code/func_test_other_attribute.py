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
def test_other_attribute(self):

    class OtherExtendedResult:

        def foo(self):
            return 2
        bar = 1
    self.result = OtherExtendedResult()
    self.make_converter()
    self.assertEqual(1, self.converter.bar)
    self.assertEqual(2, self.converter.foo())