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
def assertEqualStat(self, expected, actual):
    """assert that expected and actual are the same stat result.

        :param expected: A stat result.
        :param actual: A stat result.
        :raises AssertionError: If the expected and actual stat values differ
            other than by atime.
        """
    self.assertEqual(expected.st_size, actual.st_size, 'st_size did not match')
    self.assertEqual(expected.st_mtime, actual.st_mtime, 'st_mtime did not match')
    self.assertEqual(expected.st_ctime, actual.st_ctime, 'st_ctime did not match')
    if sys.platform == 'win32':
        self.assertEqual(0, expected.st_dev)
        self.assertEqual(0, actual.st_dev)
        self.assertEqual(0, expected.st_ino)
        self.assertEqual(0, actual.st_ino)
    else:
        self.assertEqual(expected.st_dev, actual.st_dev, 'st_dev did not match')
        self.assertEqual(expected.st_ino, actual.st_ino, 'st_ino did not match')
    self.assertEqual(expected.st_mode, actual.st_mode, 'st_mode did not match')