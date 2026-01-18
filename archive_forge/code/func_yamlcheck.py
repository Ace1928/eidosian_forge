from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def yamlcheck(python: PythonConfig, explain: bool=False) -> t.Optional[bool]:
    """Return True if PyYAML has libyaml support, False if it does not and None if it was not found."""
    stdout = raw_command([python.path, os.path.join(ANSIBLE_TEST_TARGET_TOOLS_ROOT, 'yamlcheck.py')], capture=True, explain=explain)[0]
    if explain:
        return None
    result = json.loads(stdout)
    if not result['yaml']:
        return None
    return result['cloader']