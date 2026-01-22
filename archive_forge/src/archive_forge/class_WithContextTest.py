import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class WithContextTest(testctx.TestContextTestCase):

    def test_unexported(self):
        self.assertRaisesRegex(NameError, 'undecorated not exported', testctx.context._wrap, undecorated)