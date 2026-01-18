import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
def test_qos_list_no_association(self):
    self.qos_mock.reset_mock()
    self.qos_mock.get_associations.side_effect = [[self.qos_association], exceptions.NotFound('NotFound')]
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.qos_mock.list.assert_called_with()
    self.assertEqual(self.columns, columns)
    ex_data = copy.deepcopy(self.data)
    ex_data[1] = (self.qos_specs[1].id, self.qos_specs[1].name, self.qos_specs[1].consumer, format_columns.ListColumn(None), format_columns.DictColumn(self.qos_specs[1].specs))
    self.assertCountEqual(ex_data, list(data))