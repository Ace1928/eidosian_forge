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
def test_reset_out():
    """Test '%reset out' magic"""
    _ip.run_cell("parrot = 'dead'", store_history=True)
    _ip.run_cell('parrot', store_history=True)
    assert 'dead' in [_ip.user_ns[x] for x in ('_', '__', '___')]
    _ip.run_line_magic('reset', '-f out')
    assert 'dead' not in [_ip.user_ns[x] for x in ('_', '__', '___')]
    assert len(_ip.user_ns['Out']) == 0