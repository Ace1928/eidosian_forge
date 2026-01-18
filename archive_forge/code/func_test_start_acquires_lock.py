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
def test_start_acquires_lock(self):
    context = priv_context.PrivContext('test', capabilities=[])
    context.channel = 'something not None'
    context.start_lock = mock.Mock()
    context.start_lock.__enter__ = mock.Mock()
    context.start_lock.__exit__ = mock.Mock()
    self.assertFalse(context.start_lock.__enter__.called)
    context.start()
    self.assertTrue(context.start_lock.__enter__.called)