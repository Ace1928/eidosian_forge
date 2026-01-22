import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class SeparationTest(testctx.TestContextTestCase):

    def test_getpid(self):
        priv_pid = priv_getpid()
        self.assertNotMyPid(priv_pid)

    def test_client_mode(self):
        self.assertNotMyPid(priv_getpid())
        self.addCleanup(testctx.context.set_client_mode, True)
        testctx.context.set_client_mode(False)
        self.assertEqual(os.getpid(), priv_getpid())
        testctx.context.set_client_mode(True)
        self.assertNotMyPid(priv_getpid())