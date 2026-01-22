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
class TestCaseWithTransport(TestCaseInTempDir):
    """A test case that provides get_url and get_readonly_url facilities.

    These back onto two transport servers, one for readonly access and one for
    read write access.

    If no explicit class is provided for readonly access, a
    ReadonlyTransportDecorator is used instead which allows the use of non disk
    based read write transports.

    If an explicit class is provided for readonly access, that server and the
    readwrite one must both define get_url() as resolving to os.getcwd().
    """

    def setUp(self):
        super().setUp()
        self.__vfs_server = None

    def get_vfs_only_server(self):
        """See TestCaseWithMemoryTransport.

        This is useful for some tests with specific servers that need
        diagnostics.
        """
        if self.__vfs_server is None:
            self.__vfs_server = self.vfs_transport_factory()
            self.start_server(self.__vfs_server)
        return self.__vfs_server

    def make_branch_and_tree(self, relpath, format=None):
        """Create a branch on the transport and a tree locally.

        If the transport is not a LocalTransport, the Tree can't be created on
        the transport.  In that case if the vfs_transport_factory is
        LocalURLServer the working tree is created in the local
        directory backing the transport, and the returned tree's branch and
        repository will also be accessed locally. Otherwise a lightweight
        checkout is created and returned.

        We do this because we can't physically create a tree in the local
        path, with a branch reference to the transport_factory url, and
        a branch + repository in the vfs_transport, unless the vfs_transport
        namespace is distinct from the local disk - the two branch objects
        would collide. While we could construct a tree with its branch object
        pointing at the transport_factory transport in memory, reopening it
        would behaving unexpectedly, and has in the past caused testing bugs
        when we tried to do it that way.

        :param format: The BzrDirFormat.
        :returns: the WorkingTree.
        """
        format = self.resolve_format(format=format)
        if not format.supports_workingtrees:
            b = self.make_branch(relpath + '.branch', format=format)
            return b.create_checkout(relpath, lightweight=True)
        b = self.make_branch(relpath, format=format)
        try:
            return b.controldir.create_workingtree()
        except errors.NotLocalUrl:
            if self.vfs_transport_factory is test_server.LocalURLServer:
                local_controldir = controldir.ControlDir.open(self.get_vfs_only_url(relpath))
                wt = local_controldir.create_workingtree()
                if wt.branch._format != b._format:
                    wt._branch = b
                    self.assertIs(b, wt.branch)
                return wt
            else:
                return b.create_checkout(relpath, lightweight=True)

    def assertIsDirectory(self, relpath, transport):
        """Assert that relpath within transport is a directory.

        This may not be possible on all transports; in that case it propagates
        a TransportNotPossible.
        """
        try:
            mode = transport.stat(relpath).st_mode
        except _mod_transport.NoSuchFile:
            self.fail('path %s is not a directory; no such file' % relpath)
        if not stat.S_ISDIR(mode):
            self.fail('path %s is not a directory; has mode %#o' % (relpath, mode))

    def assertTreesEqual(self, left, right):
        """Check that left and right have the same content and properties."""
        self.assertEqual(left.get_parent_ids(), right.get_parent_ids())
        differences = left.changes_from(right)
        self.assertFalse(differences.has_changed(), 'Trees {!r} and {!r} are different: {!r}'.format(left, right, differences))

    def disable_missing_extensions_warning(self):
        """Some tests expect a precise stderr content.

        There is no point in forcing them to duplicate the extension related
        warning.
        """
        config.GlobalConfig().set_user_option('suppress_warnings', 'missing_extensions')