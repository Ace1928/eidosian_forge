from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_show(self):
    expected_columns = tuple(self.share_instance._info.keys())
    expected_data_dic = tuple()
    for column in expected_columns:
        expected_data_dic += (self.share_instance._info[column],)
    expected_columns += ('export_locations',)
    expected_data_dic += (dict(self.export_locations[0]),)
    cliutils.convert_dict_list_to_string = mock.Mock()
    cliutils.convert_dict_list_to_string.return_value = dict(self.export_locations[0])
    arglist = [self.share_instance.id]
    verifylist = [('instance', self.share_instance.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.instances_mock.get.assert_called_with(self.share_instance.id)
    self.assertCountEqual(expected_columns, columns)
    self.assertCountEqual(expected_data_dic, data)