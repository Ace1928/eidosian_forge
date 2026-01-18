import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
def test_shell_args_tls_options(self):
    """Test the TLS verify and CA cert file options"""
    _shell = utils.make_shell()
    utils.fake_execute(_shell, 'module list')
    self.assertIsNone(_shell.options.verify)
    self.assertIsNone(_shell.options.insecure)
    self.assertIsNone(_shell.options.cacert)
    self.assertTrue(_shell.client_manager.verify)
    self.assertIsNone(_shell.client_manager.cacert)
    utils.fake_execute(_shell, '--verify module list')
    self.assertTrue(_shell.options.verify)
    self.assertIsNone(_shell.options.insecure)
    self.assertIsNone(_shell.options.cacert)
    self.assertTrue(_shell.client_manager.verify)
    self.assertIsNone(_shell.client_manager.cacert)
    utils.fake_execute(_shell, '--insecure module list')
    self.assertIsNone(_shell.options.verify)
    self.assertTrue(_shell.options.insecure)
    self.assertIsNone(_shell.options.cacert)
    self.assertFalse(_shell.client_manager.verify)
    self.assertIsNone(_shell.client_manager.cacert)
    utils.fake_execute(_shell, '--os-cacert foo module list')
    self.assertIsNone(_shell.options.verify)
    self.assertIsNone(_shell.options.insecure)
    self.assertEqual('foo', _shell.options.cacert)
    self.assertEqual('foo', _shell.client_manager.verify)
    self.assertEqual('foo', _shell.client_manager.cacert)
    utils.fake_execute(_shell, '--os-cacert foo --verify module list')
    self.assertTrue(_shell.options.verify)
    self.assertIsNone(_shell.options.insecure)
    self.assertEqual('foo', _shell.options.cacert)
    self.assertEqual('foo', _shell.client_manager.verify)
    self.assertEqual('foo', _shell.client_manager.cacert)
    utils.fake_execute(_shell, '--os-cacert foo --insecure module list')
    self.assertIsNone(_shell.options.verify)
    self.assertTrue(_shell.options.insecure)
    self.assertEqual('foo', _shell.options.cacert)
    self.assertFalse(_shell.client_manager.verify)
    self.assertIsNone(_shell.client_manager.cacert)