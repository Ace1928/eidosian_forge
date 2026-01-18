import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_pool_set_tag(self):
    self.api_mock.pool_show.return_value = {'tags': ['foo']}
    arglist = [self._po.id, '--tag', 'bar']
    verifylist = [('pool', self._po.id), ('tags', ['bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_set.assert_called_once()
    kwargs = self.api_mock.pool_set.mock_calls[0][2]
    tags = kwargs['json']['pool']['tags']
    self.assertEqual(2, len(tags))
    self.assertIn('foo', tags)
    self.assertIn('bar', tags)