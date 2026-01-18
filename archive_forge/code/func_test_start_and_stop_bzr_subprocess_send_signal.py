import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_start_and_stop_bzr_subprocess_send_signal(self):
    """finish_brz_subprocess raises self.failureException if the retcode is
        not the expected one.
        """
    self.disable_missing_extensions_warning()
    process = self.start_brz_subprocess(['wait-until-signalled'], skip_if_plan_to_signal=True)
    self.assertEqual(b'running\n', process.stdout.readline())
    result = self.finish_brz_subprocess(process, send_signal=signal.SIGINT, retcode=3)
    self.assertEqual(b'', result[0])
    self.assertEqual(b'brz: interrupted\n', result[1])