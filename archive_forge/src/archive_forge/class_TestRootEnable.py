from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient.osc.v1 import database_root
from troveclient.tests.osc.v1 import fakes
class TestRootEnable(TestRoot):

    def setUp(self):
        super(TestRootEnable, self).setUp()
        self.cmd = database_root.EnableDatabaseRoot(self.app, None)
        self.data = {'instance': self.fake_root.post_instance_1234_root(), 'cluster': self.fake_root.post_cls_1234_root()}
        self.columns = ('name', 'password')

    @mock.patch.object(utils, 'find_resource')
    def test_enable_instance_1234_root(self, mock_find):
        self.root_client.create_instance_root.return_value = self.data['instance']
        args = ['1234']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(('root', 'password'), data)

    @mock.patch.object(utils, 'find_resource')
    def test_enable_cluster_1234_root(self, mock_find):
        mock_find.side_effect = [exceptions.CommandError(), (None, 'cluster')]
        self.root_client.create_cluster_root.return_value = self.data['cluster']
        args = ['1234']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(('root', 'password'), data)

    @mock.patch.object(utils, 'find_resource')
    def test_enable_instance_root_with_password(self, mock_find):
        args = ['1234', '--root_password', 'secret']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.root_client.create_instance_root(None, root_password='secret')

    @mock.patch.object(utils, 'find_resource')
    def test_enable_cluster_root_with_password(self, mock_find):
        args = ['1234', '--root_password', 'secret']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.root_client.create_cluster_root(None, root_password='secret')