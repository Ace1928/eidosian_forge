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
class TestWithFakedStartBzrSubprocess(tests.TestCaseWithTransport):
    """Base class for tests testing how we might run bzr."""

    def setUp(self):
        super().setUp()
        self.subprocess_calls = []

    def start_brz_subprocess(self, process_args, env_changes=None, skip_if_plan_to_signal=False, working_dir=None, allow_plugins=False):
        """capture what run_brz_subprocess tries to do."""
        self.subprocess_calls.append({'process_args': process_args, 'env_changes': env_changes, 'skip_if_plan_to_signal': skip_if_plan_to_signal, 'working_dir': working_dir, 'allow_plugins': allow_plugins})
        return self.next_subprocess