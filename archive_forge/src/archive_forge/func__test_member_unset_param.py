import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_member_unset_param(self, param):
    self.api_mock.member_set.reset_mock()
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._mem.pool_id, self._mem.id, '--%s' % arg_param]
    ref_body = {'member': {param: None}}
    verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json=ref_body)