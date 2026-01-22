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
def _get_sample_text(self, encoding='unicode_internal'):
    if encoding is None:
        encoding = 'unicode_internal'
    for u in self._sample_texts:
        try:
            b = u.encode(encoding)
            if u == b.decode(encoding):
                return (u, u)
        except (LookupError, UnicodeError):
            pass
    self.skipTest('Could not find a sample text for encoding: %r' % encoding)