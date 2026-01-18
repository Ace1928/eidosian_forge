from unittest import mock
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.tests import fakes as test_fakes
from osc_lib.tests import utils as test_utils
def test_validate_os_beta_command_enabled(self):
    cmd = FakeCommand(mock.Mock(), mock.Mock())
    cmd.app = mock.Mock()
    cmd.app.options = test_fakes.FakeOptions()
    cmd.app.options.os_beta_command = True
    cmd.validate_os_beta_command_enabled()
    cmd.app.options.os_beta_command = False
    self.assertRaises(exceptions.CommandError, cmd.validate_os_beta_command_enabled)