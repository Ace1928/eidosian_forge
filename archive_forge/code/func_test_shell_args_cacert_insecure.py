import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def test_shell_args_cacert_insecure(self):
    _shell = shell.OpenStackShell()
    _shell.run('--os-cacert xyzpdq --insecure extension list'.split())
    self.assertNotEqual(len(self.requests_mock.request_history), 0)
    self.assertFalse(self.requests_mock.request_history[0].verify)