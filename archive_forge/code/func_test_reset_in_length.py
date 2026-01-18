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
def test_reset_in_length():
    """Test that '%reset in' preserves In[] length"""
    _ip.run_cell("print 'foo'")
    _ip.run_cell('reset -f in')
    assert len(_ip.user_ns['In']) == _ip.displayhook.prompt_count + 1