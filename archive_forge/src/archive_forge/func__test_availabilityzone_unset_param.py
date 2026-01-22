import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_availabilityzone_unset_param(self, param):
    self.api_mock.availabilityzone_set.reset_mock()
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._availabilityzone.name, '--%s' % arg_param]
    ref_body = {'availability_zone': {param: None}}
    verifylist = [('availabilityzone', self._availabilityzone.name)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    print(verifylist)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.availabilityzone_set.assert_called_once_with(self._availabilityzone.name, json=ref_body)