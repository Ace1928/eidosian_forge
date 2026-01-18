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
@dec.skipif(execution.profile is None)
def test_prun_special_syntax():
    """Test %%prun with IPython special syntax"""

    @register_line_magic
    def lmagic(line):
        ip = get_ipython()
        ip.user_ns['lmagic_out'] = line
    _ip.run_line_magic('prun', '-q %lmagic my line')
    assert _ip.user_ns['lmagic_out'] == 'my line'
    _ip.run_cell_magic('prun', '-q', '%lmagic my line2')
    assert _ip.user_ns['lmagic_out'] == 'my line2'