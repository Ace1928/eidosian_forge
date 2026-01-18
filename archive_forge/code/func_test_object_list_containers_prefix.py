import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import container
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_list_containers_prefix(self, c_mock):
    c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_3)]
    arglist = ['--prefix', 'bit']
    verifylist = [('prefix', 'bit')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'prefix': 'bit'}
    c_mock.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    datalist = ((object_fakes.container_name,), (object_fakes.container_name_3,))
    self.assertEqual(datalist, tuple(data))