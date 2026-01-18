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
@unittest.skipUnless(docutils is not None, 'needs docutils')
def test_register_invalid_long_description(self):
    description = ':funkie:`str`'
    metadata = {'url': 'xxx', 'author': 'xxx', 'author_email': 'xxx', 'name': 'xxx', 'version': 'xxx', 'long_description': description}
    cmd = self._get_cmd(metadata)
    cmd.ensure_finalized()
    cmd.strict = True
    inputs = Inputs('2', 'tarek', 'tarek@ziade.org')
    register_module.input = inputs
    self.addCleanup(delattr, register_module, 'input')
    self.assertRaises(DistutilsSetupError, cmd.run)