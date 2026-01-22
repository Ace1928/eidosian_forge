import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestMappingDelete(TestMapping):

    def setUp(self):
        super(TestMappingDelete, self).setUp()
        self.mapping_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.MAPPING_RESPONSE), loaded=True)
        self.mapping_mock.delete.return_value = None
        self.cmd = mapping.DeleteMapping(self.app, None)

    def test_delete_mapping(self):
        arglist = [identity_fakes.mapping_id]
        verifylist = [('mapping', [identity_fakes.mapping_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mapping_mock.delete.assert_called_with(identity_fakes.mapping_id)
        self.assertIsNone(result)