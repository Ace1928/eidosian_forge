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
class TestCaseInTempDir(TestCaseWithMemoryTransport):
    """Derived class that runs a test within a temporary directory.

    This is useful for tests that need to create a branch, etc.

    The directory is created in a slightly complex way: for each
    Python invocation, a new temporary top-level directory is created.
    All test cases create their own directory within that.  If the
    tests complete successfully, the directory is removed.

    :ivar test_base_dir: The path of the top-level directory for this
    test, which contains a home directory and a work directory.

    :ivar test_home_dir: An initially empty directory under test_base_dir
    which is used as $HOME for this test.

    :ivar test_dir: A directory under test_base_dir used as the current
    directory when the test proper is run.
    """
    OVERRIDE_PYTHON = 'python'

    def setUp(self):
        super().setUp()
        self.overrideEnv('BRZ_LOG', None)

    def check_file_contents(self, filename, expect):
        self.log('check contents of file %s' % filename)
        with open(filename, 'rb') as f:
            contents = f.read()
        if contents != expect:
            self.log('expected: %r' % expect)
            self.log('actually: %r' % contents)
            self.fail('contents of %s not as expected' % filename)

    def _getTestDirPrefix(self):
        if sys.platform in ('win32', 'cygwin'):
            name_prefix = re.sub('[<>*=+",:;_/\\-]', '_', self.id())
            name_prefix = name_prefix[-30:]
        else:
            name_prefix = re.sub('[/]', '_', self.id())
        return name_prefix

    def makeAndChdirToTestDir(self):
        """See TestCaseWithMemoryTransport.makeAndChdirToTestDir().

        For TestCaseInTempDir we create a temporary directory based on the test
        name and then create two subdirs - test and home under it.
        """
        name_prefix = osutils.pathjoin(TestCaseWithMemoryTransport.TEST_ROOT, self._getTestDirPrefix())
        name = name_prefix
        for i in range(100):
            if os.path.exists(name):
                name = name_prefix + '_' + str(i)
            else:
                self.test_base_dir = name
                self.addCleanup(self.deleteTestDir)
                os.mkdir(self.test_base_dir)
                break
        self.permit_dir(self.test_base_dir)
        self.test_home_dir = self.test_base_dir + '/home'
        os.mkdir(self.test_home_dir)
        self.test_dir = self.test_base_dir + '/work'
        os.mkdir(self.test_dir)
        os.chdir(self.test_dir)
        with open(self.test_base_dir + '/name', 'w') as f:
            f.write(self.id())

    def deleteTestDir(self):
        os.chdir(TestCaseWithMemoryTransport.TEST_ROOT)
        _rmtree_temp_dir(self.test_base_dir, test_id=self.id())

    def build_tree(self, shape, line_endings='binary', transport=None):
        """Build a test tree according to a pattern.

        shape is a sequence of file specifications.  If the final
        character is '/', a directory is created.

        This assumes that all the elements in the tree being built are new.

        This doesn't add anything to a branch.

        :type shape:    list or tuple.
        :param line_endings: Either 'binary' or 'native'
            in binary mode, exact contents are written in native mode, the
            line endings match the default platform endings.
        :param transport: A transport to write to, for building trees on VFS's.
            If the transport is readonly or None, "." is opened automatically.
        :return: None
        """
        if type(shape) not in (list, tuple):
            raise AssertionError("Parameter 'shape' should be a list or a tuple. Got %r instead" % (shape,))
        if transport is None or transport.is_readonly():
            transport = _mod_transport.get_transport_from_path('.')
        for name in shape:
            self.assertIsInstance(name, str)
            if name[-1] == '/':
                transport.mkdir(urlutils.escape(name[:-1]))
            else:
                if line_endings == 'binary':
                    end = b'\n'
                elif line_endings == 'native':
                    end = os.linesep.encode('ascii')
                else:
                    raise errors.BzrError('Invalid line ending request %r' % line_endings)
                content = b'contents of %s%s' % (name.encode('utf-8'), end)
                transport.put_bytes_non_atomic(urlutils.escape(name), content)
    build_tree_contents = staticmethod(treeshape.build_tree_contents)

    def assertInWorkingTree(self, path, root_path='.', tree=None):
        """Assert whether path or paths are in the WorkingTree"""
        if tree is None:
            tree = workingtree.WorkingTree.open(root_path)
        if not isinstance(path, str):
            for p in path:
                self.assertInWorkingTree(p, tree=tree)
        else:
            self.assertTrue(tree.is_versioned(path), path + ' not in working tree.')

    def assertNotInWorkingTree(self, path, root_path='.', tree=None):
        """Assert whether path or paths are not in the WorkingTree"""
        if tree is None:
            tree = workingtree.WorkingTree.open(root_path)
        if not isinstance(path, str):
            for p in path:
                self.assertNotInWorkingTree(p, tree=tree)
        else:
            self.assertFalse(tree.is_versioned(path), path + ' in working tree.')