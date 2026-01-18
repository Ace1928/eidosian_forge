import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase

        L{twisted.python.compat._get_async_param} raises a deprecation
        warning if async keyword argument is passed.
        