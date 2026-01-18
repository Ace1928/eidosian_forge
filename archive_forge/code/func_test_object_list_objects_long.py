import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import object as obj
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_list_objects_long(self, o_mock):
    o_mock.return_value = [copy.deepcopy(object_fakes.OBJECT), copy.deepcopy(object_fakes.OBJECT_2)]
    arglist = ['--long', object_fakes.container_name]
    verifylist = [('long', True), ('container', object_fakes.container_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {}
    o_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
    collist = ('Name', 'Bytes', 'Hash', 'Content Type', 'Last Modified')
    self.assertEqual(collist, columns)
    datalist = ((object_fakes.object_name_1, object_fakes.object_bytes_1, object_fakes.object_hash_1, object_fakes.object_content_type_1, object_fakes.object_modified_1), (object_fakes.object_name_2, object_fakes.object_bytes_2, object_fakes.object_hash_2, object_fakes.object_content_type_2, object_fakes.object_modified_2))
    self.assertEqual(datalist, tuple(data))