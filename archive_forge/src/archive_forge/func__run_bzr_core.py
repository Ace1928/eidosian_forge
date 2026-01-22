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
def _run_bzr_core(self, args, encoding, stdin, stdout, stderr, working_dir):
    chk_map.clear_cache()
    self.log('run brz: %r', args)
    self._last_cmd_stdout = stdout
    self._last_cmd_stderr = stderr
    old_ui_factory = ui.ui_factory
    ui.ui_factory = ui_testing.TestUIFactory(stdin=stdin, stdout=self._last_cmd_stdout, stderr=self._last_cmd_stderr)
    cwd = None
    if working_dir is not None:
        cwd = osutils.getcwd()
        os.chdir(working_dir)
    try:
        with ui.ui_factory:
            result = self.apply_redirected(ui.ui_factory.stdin, stdout, stderr, _mod_commands.run_bzr_catch_user_errors, args)
    finally:
        ui.ui_factory = old_ui_factory
        if cwd is not None:
            os.chdir(cwd)
    return result