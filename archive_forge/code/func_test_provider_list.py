import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import provider
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_provider_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.provider_list.assert_called_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))