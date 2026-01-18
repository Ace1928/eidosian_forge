import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import provider
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_provider_capability_list_all(self):
    arglist = ['provider1']
    verifylist = [('provider', 'provider1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    capabilities = list(result[1])
    self.api_mock.provider_flavor_capability_list.assert_called_with(provider='provider1')
    self.api_mock.provider_availability_zone_capability_list.assert_called_with(provider='provider1')
    self.assertIn(tuple(['flavor'] + list(attr_consts.CAPABILITY_ATTRS.values())), capabilities)
    self.assertIn(tuple(['availability_zone'] + list(attr_consts.CAPABILITY_ATTRS.values())), capabilities)