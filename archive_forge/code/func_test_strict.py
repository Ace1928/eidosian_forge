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
def test_strict(self):
    cmd = self._get_cmd({})
    cmd.ensure_finalized()
    cmd.strict = 1
    self.assertRaises(DistutilsSetupError, cmd.run)
    metadata = {'url': 'xxx', 'author': 'xxx', 'author_email': 'éxéxé', 'name': 'xxx', 'version': 'xxx', 'long_description': 'title\n==\n\ntext'}
    cmd = self._get_cmd(metadata)
    cmd.ensure_finalized()
    cmd.strict = 1
    self.assertRaises(DistutilsSetupError, cmd.run)
    metadata['long_description'] = 'title\n=====\n\ntext'
    cmd = self._get_cmd(metadata)
    cmd.ensure_finalized()
    cmd.strict = 1
    inputs = Inputs('1', 'tarek', 'y')
    register_module.input = inputs.__call__
    try:
        cmd.run()
    finally:
        del register_module.input
    cmd = self._get_cmd()
    cmd.ensure_finalized()
    inputs = Inputs('1', 'tarek', 'y')
    register_module.input = inputs.__call__
    try:
        cmd.run()
    finally:
        del register_module.input
    metadata = {'url': 'xxx', 'author': 'Éric', 'author_email': 'xxx', 'name': 'xxx', 'version': 'xxx', 'description': 'Something about esszet ß', 'long_description': 'More things about esszet ß'}
    cmd = self._get_cmd(metadata)
    cmd.ensure_finalized()
    cmd.strict = 1
    inputs = Inputs('1', 'tarek', 'y')
    register_module.input = inputs.__call__
    try:
        cmd.run()
    finally:
        del register_module.input