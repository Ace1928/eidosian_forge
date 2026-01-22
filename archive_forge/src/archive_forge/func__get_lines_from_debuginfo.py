from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def _get_lines_from_debuginfo(self, metadata):
    md_def_map = self._get_metadata_map(metadata)
    lines = set()
    for md in md_def_map.values():
        m = re.match('!DILocation\\(line: (\\d+),', md)
        if m:
            ln = int(m.group(1))
            lines.add(ln)
    return lines