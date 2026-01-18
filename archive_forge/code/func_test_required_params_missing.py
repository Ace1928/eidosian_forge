from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
def test_required_params_missing(self):
    args = [self.random_name('backup')]
    parsed_args = self.check_parser(self.cmd, args, [])
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)