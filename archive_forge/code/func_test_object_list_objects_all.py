import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import object as obj
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_list_objects_all(self, o_mock):
    o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT), copy.deepcopy(object_fakes.OBJECT_2)]
    arglist = ['--all', object_fakes.container_name]
    verifylist = [('all', True), ('container', object_fakes.container_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'full_listing': True}
    o_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
    self.assertEqual(self.columns, columns)
    datalist = ((object_fakes.object_name_1,), (object_fakes.object_name_2,))
    self.assertEqual(datalist, tuple(data))