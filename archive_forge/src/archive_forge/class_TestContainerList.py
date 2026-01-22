import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import container
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
@mock.patch('openstackclient.api.object_store_v1.APIv1.container_list')
class TestContainerList(TestContainer):

    def setUp(self):
        super(TestContainerList, self).setUp()
        self.cmd = container.ListContainer(self.app, None)

    def test_object_list_containers_no_options(self, c_mock):
        c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_3), copy.deepcopy(object_fakes.CONTAINER_2)]
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {}
        c_mock.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.container_name,), (object_fakes.container_name_3,), (object_fakes.container_name_2,))
        self.assertEqual(datalist, tuple(data))

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

    def test_object_list_containers_marker(self, c_mock):
        c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_3)]
        arglist = ['--marker', object_fakes.container_name, '--end-marker', object_fakes.container_name_3]
        verifylist = [('marker', object_fakes.container_name), ('end_marker', object_fakes.container_name_3)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': object_fakes.container_name, 'end_marker': object_fakes.container_name_3}
        c_mock.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.container_name,), (object_fakes.container_name_3,))
        self.assertEqual(datalist, tuple(data))

    def test_object_list_containers_limit(self, c_mock):
        c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_3)]
        arglist = ['--limit', '2']
        verifylist = [('limit', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'limit': 2}
        c_mock.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.container_name,), (object_fakes.container_name_3,))
        self.assertEqual(datalist, tuple(data))

    def test_object_list_containers_long(self, c_mock):
        c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_3)]
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {}
        c_mock.assert_called_with(**kwargs)
        collist = ('Name', 'Bytes', 'Count')
        self.assertEqual(collist, columns)
        datalist = ((object_fakes.container_name, object_fakes.container_bytes, object_fakes.container_count), (object_fakes.container_name_3, object_fakes.container_bytes * 3, object_fakes.container_count * 3))
        self.assertEqual(datalist, tuple(data))

    def test_object_list_containers_all(self, c_mock):
        c_mock.return_value = [copy.deepcopy(object_fakes.CONTAINER), copy.deepcopy(object_fakes.CONTAINER_2), copy.deepcopy(object_fakes.CONTAINER_3)]
        arglist = ['--all']
        verifylist = [('all', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'full_listing': True}
        c_mock.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        datalist = ((object_fakes.container_name,), (object_fakes.container_name_2,), (object_fakes.container_name_3,))
        self.assertEqual(datalist, tuple(data))