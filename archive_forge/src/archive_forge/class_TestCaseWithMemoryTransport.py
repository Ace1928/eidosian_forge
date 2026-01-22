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
class TestCaseWithMemoryTransport(TestCase):
    """Common test class for tests that do not need disk resources.

    Tests that need disk resources should derive from TestCaseInTempDir
    orTestCaseWithTransport.

    TestCaseWithMemoryTransport sets the TEST_ROOT variable for all brz tests.

    For TestCaseWithMemoryTransport the ``test_home_dir`` is set to the name of
    a directory which does not exist. This serves to help ensure test isolation
    is preserved. ``test_dir`` is set to the TEST_ROOT, as is cwd, because they
    must exist. However, TestCaseWithMemoryTransport does not offer local file
    defaults for the transport in tests, nor does it obey the command line
    override, so tests that accidentally write to the common directory should
    be rare.

    :cvar TEST_ROOT: Directory containing all temporary directories, plus a
        ``.bzr`` directory that stops us ascending higher into the filesystem.
    """
    TEST_ROOT = None
    _TEST_NAME = 'test'

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.vfs_transport_factory = default_transport
        self.transport_server = None
        self.transport_readonly_server = None
        self.__vfs_server = None

    def setUp(self):
        super().setUp()

        def _add_disconnect_cleanup(transport):
            """Schedule disconnection of given transport at test cleanup

            This needs to happen for all connected transports or leaks occur.

            Note reconnections may mean we call disconnect multiple times per
            transport which is suboptimal but seems harmless.
            """
            self.addCleanup(transport.disconnect)
        _mod_transport.Transport.hooks.install_named_hook('post_connect', _add_disconnect_cleanup, None)
        self._make_test_root()
        self.addCleanup(os.chdir, osutils.getcwd())
        self.makeAndChdirToTestDir()
        self.overrideEnvironmentForTesting()
        self.__readonly_server = None
        self.__server = None
        self.reduceLockdirTimeout()
        self.overrideAttr(config, '_shared_stores', {})

    def get_transport(self, relpath=None):
        """Return a writeable transport.

        This transport is for the test scratch space relative to
        "self._test_root"

        :param relpath: a path relative to the base url.
        """
        t = _mod_transport.get_transport_from_url(self.get_url(relpath))
        self.assertFalse(t.is_readonly())
        return t

    def get_readonly_transport(self, relpath=None):
        """Return a readonly transport for the test scratch space

        This can be used to test that operations which should only need
        readonly access in fact do not try to write.

        :param relpath: a path relative to the base url.
        """
        t = _mod_transport.get_transport_from_url(self.get_readonly_url(relpath))
        self.assertTrue(t.is_readonly())
        return t

    def create_transport_readonly_server(self):
        """Create a transport server from class defined at init.

        This is mostly a hook for daughter classes.
        """
        return self.transport_readonly_server()

    def get_readonly_server(self):
        """Get the server instance for the readonly transport

        This is useful for some tests with specific servers to do diagnostics.
        """
        if self.__readonly_server is None:
            if self.transport_readonly_server is None:
                self.__readonly_server = test_server.ReadonlyServer()
            else:
                self.__readonly_server = self.create_transport_readonly_server()
            self.start_server(self.__readonly_server, self.get_vfs_only_server())
        return self.__readonly_server

    def get_readonly_url(self, relpath=None):
        """Get a URL for the readonly transport.

        This will either be backed by '.' or a decorator to the transport
        used by self.get_url()
        relpath provides for clients to get a path relative to the base url.
        These should only be downwards relative, not upwards.
        """
        base = self.get_readonly_server().get_url()
        return self._adjust_url(base, relpath)

    def get_vfs_only_server(self):
        """Get the vfs only read/write server instance.

        This is useful for some tests with specific servers that need
        diagnostics.

        For TestCaseWithMemoryTransport this is always a MemoryServer, and there
        is no means to override it.
        """
        if self.__vfs_server is None:
            self.__vfs_server = memory.MemoryServer()
            self.start_server(self.__vfs_server)
        return self.__vfs_server

    def get_server(self):
        """Get the read/write server instance.

        This is useful for some tests with specific servers that need
        diagnostics.

        This is built from the self.transport_server factory. If that is None,
        then the self.get_vfs_server is returned.
        """
        if self.__server is None:
            if self.transport_server is None or self.transport_server is self.vfs_transport_factory:
                self.__server = self.get_vfs_only_server()
            else:
                self.__server = self.transport_server()
                self.start_server(self.__server, self.get_vfs_only_server())
        return self.__server

    def _adjust_url(self, base, relpath):
        """Get a URL (or maybe a path) for the readwrite transport.

        This will either be backed by '.' or to an equivalent non-file based
        facility.
        relpath provides for clients to get a path relative to the base url.
        These should only be downwards relative, not upwards.
        """
        if relpath is not None and relpath != '.':
            if not base.endswith('/'):
                base = base + '/'
            if base.startswith('./') or base.startswith('/'):
                base += relpath
            else:
                base += urlutils.escape(relpath)
        return base

    def get_url(self, relpath=None):
        """Get a URL (or maybe a path) for the readwrite transport.

        This will either be backed by '.' or to an equivalent non-file based
        facility.
        relpath provides for clients to get a path relative to the base url.
        These should only be downwards relative, not upwards.
        """
        base = self.get_server().get_url()
        return self._adjust_url(base, relpath)

    def get_vfs_only_url(self, relpath=None):
        """Get a URL (or maybe a path for the plain old vfs transport.

        This will never be a smart protocol.  It always has all the
        capabilities of the local filesystem, but it might actually be a
        MemoryTransport or some other similar virtual filesystem.

        This is the backing transport (if any) of the server returned by
        get_url and get_readonly_url.

        :param relpath: provides for clients to get a path relative to the base
            url.  These should only be downwards relative, not upwards.
        :return: A URL
        """
        base = self.get_vfs_only_server().get_url()
        return self._adjust_url(base, relpath)

    def _create_safety_net(self):
        """Make a fake bzr directory.

        This prevents any tests propagating up onto the TEST_ROOT directory's
        real branch.
        """
        root = TestCaseWithMemoryTransport.TEST_ROOT
        try:
            self.assertIs(None, os.environ.get('BRZ_HOME', None))
            os.environ['BRZ_HOME'] = root
            from breezy.bzr.bzrdir import BzrDirMetaFormat1
            wt = controldir.ControlDir.create_standalone_workingtree(root, format=BzrDirMetaFormat1())
            del os.environ['BRZ_HOME']
        except Exception as e:
            self.fail('Fail to initialize the safety net: {!r}\n'.format(e))
        TestCaseWithMemoryTransport._SAFETY_NET_PRISTINE_DIRSTATE = wt.control_transport.get_bytes('dirstate')

    def _check_safety_net(self):
        """Check that the safety .bzr directory have not been touched.

        _make_test_root have created a .bzr directory to prevent tests from
        propagating. This method ensures than a test did not leaked.
        """
        root = TestCaseWithMemoryTransport.TEST_ROOT
        t = _mod_transport.get_transport_from_path(root)
        self.permit_url(t.base)
        if t.get_bytes('.bzr/checkout/dirstate') != TestCaseWithMemoryTransport._SAFETY_NET_PRISTINE_DIRSTATE:
            _rmtree_temp_dir(root + '/.bzr', test_id=self.id())
            self._create_safety_net()
            raise AssertionError('%s/.bzr should not be modified' % root)

    def _make_test_root(self):
        if TestCaseWithMemoryTransport.TEST_ROOT is None:
            root = osutils.realpath(tempfile.mkdtemp(prefix='testbzr-', suffix='.tmp'))
            TestCaseWithMemoryTransport.TEST_ROOT = root
            self._create_safety_net()
            atexit.register(_rmtree_temp_dir, root)
        self.permit_dir(TestCaseWithMemoryTransport.TEST_ROOT)
        self.addCleanup(self._check_safety_net)

    def makeAndChdirToTestDir(self):
        """Create a temporary directories for this one test.

        This must set self.test_home_dir and self.test_dir and chdir to
        self.test_dir.

        For TestCaseWithMemoryTransport we chdir to the TEST_ROOT for this
        test.
        """
        os.chdir(TestCaseWithMemoryTransport.TEST_ROOT)
        self.test_dir = TestCaseWithMemoryTransport.TEST_ROOT
        self.test_home_dir = self.test_dir + '/MemoryTransportMissingHomeDir'
        self.permit_dir(self.test_dir)

    def make_branch(self, relpath, format=None, name=None):
        """Create a branch on the transport at relpath."""
        repo = self.make_repository(relpath, format=format)
        return repo.controldir.create_branch(append_revisions_only=False, name=name)

    def get_default_format(self):
        return 'default'

    def resolve_format(self, format):
        """Resolve an object to a ControlDir format object.

        The initial format object can either already be
        a ControlDirFormat, None (for the default format),
        or a string with the name of the control dir format.

        :param format: Object to resolve
        :return A ControlDirFormat instance
        """
        if format is None:
            format = self.get_default_format()
        if isinstance(format, str):
            format = controldir.format_registry.make_controldir(format)
        return format

    def make_controldir(self, relpath, format=None):
        try:
            maybe_a_url = self.get_url(relpath)
            segments = maybe_a_url.rsplit('/', 1)
            t = _mod_transport.get_transport(maybe_a_url)
            if len(segments) > 1 and segments[-1] not in ('', '.'):
                t.ensure_base()
            format = self.resolve_format(format)
            return format.initialize_on_transport(t)
        except errors.UninitializableFormat:
            raise TestSkipped('Format %s is not initializable.' % format)

    def make_repository(self, relpath, shared=None, format=None):
        """Create a repository on our default transport at relpath.

        Note that relpath must be a relative path, not a full url.
        """
        made_control = self.make_controldir(relpath, format=format)
        return made_control.create_repository(shared=shared)

    def make_smart_server(self, path, backing_server=None):
        if backing_server is None:
            backing_server = self.get_server()
        smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(smart_server, backing_server)
        remote_transport = _mod_transport.get_transport_from_url(smart_server.get_url()).clone(path)
        return remote_transport

    def make_branch_and_memory_tree(self, relpath, format=None):
        """Create a branch on the default transport and a MemoryTree for it."""
        b = self.make_branch(relpath, format=format)
        return b.create_memorytree()

    def make_branch_builder(self, relpath, format=None):
        branch = self.make_branch(relpath, format=format)
        return branchbuilder.BranchBuilder(branch=branch)

    def overrideEnvironmentForTesting(self):
        test_home_dir = self.test_home_dir
        self.overrideEnv('HOME', test_home_dir)
        self.overrideEnv('BRZ_HOME', test_home_dir)
        self.overrideEnv('GNUPGHOME', os.path.join(test_home_dir, '.gnupg'))

    def setup_smart_server_with_call_log(self):
        """Sets up a smart server as the transport server with a call log."""
        self.transport_server = test_server.SmartTCPServer_for_testing
        self.hpss_connections = []
        self.hpss_calls = []
        import traceback
        prefix_length = len(traceback.extract_stack()) - 2

        def capture_hpss_call(params):
            self.hpss_calls.append(CapturedCall(params, prefix_length))

        def capture_connect(transport):
            self.hpss_connections.append(transport)
        client._SmartClient.hooks.install_named_hook('call', capture_hpss_call, None)
        _mod_transport.Transport.hooks.install_named_hook('post_connect', capture_connect, None)

    def reset_smart_call_log(self):
        self.hpss_calls = []
        self.hpss_connections = []