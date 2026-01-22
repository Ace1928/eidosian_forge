import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
class TestXdel(tt.TempFileMixin):

    def test_xdel(self):
        """Test that references from %run are cleared by xdel."""
        src = 'class A(object):\n    monitor = []\n    def __del__(self):\n        self.monitor.append(1)\na = A()\n'
        self.mktmp(src)
        _ip.run_line_magic('run', '%s' % self.fname)
        _ip.run_cell('a')
        monitor = _ip.user_ns['A'].monitor
        assert monitor == []
        _ip.run_line_magic('xdel', 'a')
        gc.collect(0)
        assert monitor == [1]