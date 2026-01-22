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
def _inject_stream_into_subunit(self, stream):
    """Monkey-patch subunit so the extra output goes to stream not stdout

        Some APIs need rewriting so this kind of bogus hackery can be replaced
        by passing the stream param from run_tests down into ProtocolTestCase.
        """
    from subunit import ProtocolTestCase
    _original_init = ProtocolTestCase.__init__

    def _init_with_passthrough(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self._passthrough = stream
    self.overrideAttr(ProtocolTestCase, '__init__', _init_with_passthrough)