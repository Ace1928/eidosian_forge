import copy
from unittest import mock
from openstackclient.api import object_store_v1 as object_store
from openstackclient.object.v1 import container
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
@mock.patch('openstackclient.api.object_store_v1.APIv1.object_delete')
@mock.patch('openstackclient.api.object_store_v1.APIv1.object_list')
@mock.patch('openstackclient.api.object_store_v1.APIv1.container_delete')
class TestContainerDelete(TestContainer):

    def setUp(self):
        super(TestContainerDelete, self).setUp()
        self.cmd = container.DeleteContainer(self.app, None)

    def test_container_delete(self, c_mock, o_list_mock, o_delete_mock):
        c_mock.return_value = None
        arglist = [object_fakes.container_name]
        verifylist = [('containers', [object_fakes.container_name]), ('recursive', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertIsNone(self.cmd.take_action(parsed_args))
        kwargs = {}
        c_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
        self.assertFalse(o_list_mock.called)
        self.assertFalse(o_delete_mock.called)

    def test_recursive_delete(self, c_mock, o_list_mock, o_delete_mock):
        c_mock.return_value = None
        o_list_mock.return_value = [object_fakes.OBJECT]
        o_delete_mock.return_value = None
        arglist = ['--recursive', object_fakes.container_name]
        verifylist = [('containers', [object_fakes.container_name]), ('recursive', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertIsNone(self.cmd.take_action(parsed_args))
        kwargs = {}
        c_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
        o_list_mock.assert_called_with(container=object_fakes.container_name)
        o_delete_mock.assert_called_with(container=object_fakes.container_name, object=object_fakes.OBJECT['name'])

    def test_r_delete(self, c_mock, o_list_mock, o_delete_mock):
        c_mock.return_value = None
        o_list_mock.return_value = [object_fakes.OBJECT]
        o_delete_mock.return_value = None
        arglist = ['-r', object_fakes.container_name]
        verifylist = [('containers', [object_fakes.container_name]), ('recursive', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertIsNone(self.cmd.take_action(parsed_args))
        kwargs = {}
        c_mock.assert_called_with(container=object_fakes.container_name, **kwargs)
        o_list_mock.assert_called_with(container=object_fakes.container_name)
        o_delete_mock.assert_called_with(container=object_fakes.container_name, object=object_fakes.OBJECT['name'])