import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_listener_unset_param(self, param):
    self.api_mock.listener_set.reset_mock()
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._listener.id, '--%s' % arg_param]
    if param == 'default_pool':
        param = 'default_pool_id'
    ref_body = {'listener': {param: None}}
    verifylist = [('listener', self._listener.id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_set.assert_called_once_with(self._listener.id, json=ref_body)