import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def test_shell_args_verify(self):
    _shell = shell.OpenStackShell()
    _shell.run('--verify extension list'.split())
    self.assertNotEqual(len(self.requests_mock.request_history), 0)
    self.assertTrue(self.requests_mock.request_history[0].verify)