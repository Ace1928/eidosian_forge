import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def _test_load_balancer_unset_param(self, param):
    self.api_mock.load_balancer_set.reset_mock()
    ref_body = {'loadbalancer': {param: None}}
    arg_param = param.replace('_', '-') if '_' in param else param
    arglist = [self._lb.id, '--%s' % arg_param]
    verifylist = [('loadbalancer', self._lb.id)]
    for ref_param in self.PARAMETERS:
        verifylist.append((ref_param, param == ref_param))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_set.assert_called_once_with(self._lb.id, json=ref_body)