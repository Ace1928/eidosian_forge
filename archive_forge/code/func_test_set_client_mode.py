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
@mock.patch.object(priv_context, 'sys')
def test_set_client_mode(self, mock_sys):
    context = priv_context.PrivContext('test', capabilities=[])
    self.assertTrue(context.client_mode)
    context.set_client_mode(False)
    self.assertFalse(context.client_mode)
    mock_sys.platform = 'win32'
    self.assertRaises(RuntimeError, context.set_client_mode, True)