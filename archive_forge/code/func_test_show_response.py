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
def test_show_response(self):
    cmd = self._get_cmd()
    inputs = Inputs('1', 'tarek', 'y')
    register_module.input = inputs.__call__
    cmd.show_response = 1
    try:
        cmd.run()
    finally:
        del register_module.input
    results = self.get_logs(INFO)
    self.assertEqual(results[3], 75 * '-' + '\nxxx\n' + 75 * '-')