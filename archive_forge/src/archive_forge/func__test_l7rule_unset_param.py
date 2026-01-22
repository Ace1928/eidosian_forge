import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_l7rule_unset_param(self, param):
    self.api_mock.l7rule_set.reset_mock()
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._l7po.id, self._l7ru.id, '--%s' % arg_param]
    ref_body = {'rule': {param: None}}
    verifylist = [('l7rule_id', self._l7ru.id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json=ref_body)