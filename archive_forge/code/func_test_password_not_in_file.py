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
def test_password_not_in_file(self):
    self.write_file(self.rc, PYPIRC_NOPASSWORD)
    cmd = self._get_cmd()
    cmd._set_config()
    cmd.finalize_options()
    cmd.send_metadata()
    self.assertEqual(cmd.distribution.password, 'password')