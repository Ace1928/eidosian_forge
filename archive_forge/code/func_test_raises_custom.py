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
def test_raises_custom(self):
    exc = self.assertRaises(CustomError, fail, custom=True)
    self.assertEqual(exc.code, 42)
    self.assertEqual(exc.msg, 'omg!')