import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def apply_redirected(self, stdin=None, stdout=None, stderr=None, a_callable=None, *args, **kwargs):
    """Call callable with redirected std io pipes.

        Returns the return code."""
    if not callable(a_callable):
        raise ValueError('a_callable must be callable.')
    if stdin is None:
        stdin = BytesIO(b'')
    if stdout is None:
        if getattr(self, '_log_file', None) is not None:
            stdout = self._log_file
        else:
            stdout = StringIO()
    if stderr is None:
        if getattr(self, '_log_file', None is not None):
            stderr = self._log_file
        else:
            stderr = StringIO()
    real_stdin = sys.stdin
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        sys.stdin = stdin
        return a_callable(*args, **kwargs)
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        sys.stdin = real_stdin