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
def _run_external_case(self):
    """Run the prepared test case in a separate module"""
    sys.path.insert(0, self.dir)
    self.addCleanup(sys.path.remove, self.dir)
    module = __import__(self.modname)
    self.addCleanup(sys.modules.pop, self.modname)
    stream = io.StringIO()
    self._run(stream, module.Test())
    return stream.getvalue()