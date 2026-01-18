from unittest import mock
from magnumclient.osc.v1 import mservices
from magnumclient.tests.osc.unit.v1 import fakes
def test_service_list(self):
    arglist = []
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.mservices_mock.list.assert_called_with()
    self.assertEqual(self.columns, columns)