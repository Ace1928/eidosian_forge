import os
import unittest
import getpass
import urllib
import warnings
from test.support.warnings_helper import check_warnings
from distutils.command import register as register_module
from distutils.command.register import register
from distutils.errors import DistutilsSetupError
from distutils.log import INFO
from distutils.tests.test_config import BasePyPIRCCommandTestCase
def test_registering(self):
    cmd = self._get_cmd()
    inputs = Inputs('2', 'tarek', 'tarek@ziade.org')
    register_module.input = inputs.__call__
    try:
        cmd.run()
    finally:
        del register_module.input
    self.assertEqual(len(self.conn.reqs), 1)
    req = self.conn.reqs[0]
    headers = dict(req.headers)
    self.assertEqual(headers['Content-length'], '608')
    self.assertIn(b'tarek', req.data)