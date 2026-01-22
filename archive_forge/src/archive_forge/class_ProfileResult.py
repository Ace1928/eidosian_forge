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
class ProfileResult(testtools.ExtendedToOriginalDecorator):
    """Generate profiling data for all activity between start and success.

    The profile data is appended to the test's _benchcalls attribute and can
    be accessed by the forwarded-to TestResult.

    While it might be cleaner do accumulate this in stopTest, addSuccess is
    where our existing output support for lsprof is, and this class aims to
    fit in with that: while it could be moved it's not necessary to accomplish
    test profiling, nor would it be dramatically cleaner.
    """

    def startTest(self, test):
        self.profiler = breezy.lsprof.BzrProfiler()
        breezy.lsprof.BzrProfiler.profiler_block = 0
        self.profiler.start()
        testtools.ExtendedToOriginalDecorator.startTest(self, test)

    def addSuccess(self, test):
        stats = self.profiler.stop()
        try:
            calls = test._benchcalls
        except AttributeError:
            test._benchcalls = []
            calls = test._benchcalls
        calls.append(((test.id(), '', ''), stats))
        testtools.ExtendedToOriginalDecorator.addSuccess(self, test)

    def stopTest(self, test):
        testtools.ExtendedToOriginalDecorator.stopTest(self, test)
        self.profiler = None