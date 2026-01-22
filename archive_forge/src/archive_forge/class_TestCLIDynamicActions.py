from unittest import mock
from mistralclient.api.v2 import dynamic_actions
from mistralclient.commands.v2 import dynamic_actions as dyn_actions_cmd
from mistralclient.tests.unit import base
class TestCLIDynamicActions(base.BaseCommandTest):

    @mock.patch('argparse.open', create=True)
    def test_create(self, mock_open):
        self.client.dynamic_actions.create.return_value = DYN_ACTION
        result = self.call(dyn_actions_cmd.Create, app_args=['my_action', 'MyAction', 'my_module'])
        self.assertEqual(('1-2-3-4', 'my_action', 'MyAction', '2-3-4-5', 'my_module', '12345', 'private', '1', '1'), result[1])

    @mock.patch('argparse.open', create=True)
    def test_create_public(self, mock_open):
        dyn_action_dict = DYN_ACTION_DICT.copy()
        dyn_action_dict['scope'] = 'public'
        dyn_action = dynamic_actions.DynamicAction(mock, dyn_action_dict)
        self.client.dynamic_actions.create.return_value = dyn_action
        result = self.call(dyn_actions_cmd.Create, app_args=['my_action', 'MyAction', 'my_module', '--public'])
        self.assertEqual(('1-2-3-4', 'my_action', 'MyAction', '2-3-4-5', 'my_module', '12345', 'public', '1', '1'), result[1])
        self.assertEqual('public', self.client.dynamic_actions.create.call_args[1]['scope'])

    @mock.patch('argparse.open', create=True)
    def test_update(self, mock_open):
        dyn_action_dict = DYN_ACTION_DICT.copy()
        dyn_action_dict['class_name'] = 'MyNewAction'
        dyn_action = dynamic_actions.DynamicAction(mock, dyn_action_dict)
        self.client.dynamic_actions.update.return_value = dyn_action
        result = self.call(dyn_actions_cmd.Update, app_args=['my_action', '--class-name', 'MyNewAction'])
        self.assertEqual(('1-2-3-4', 'my_action', 'MyNewAction', '2-3-4-5', 'my_module', '12345', 'private', '1', '1'), result[1])

    @mock.patch('argparse.open', create=True)
    def test_update_with_id(self, mock_open):
        dyn_action_dict = DYN_ACTION_DICT.copy()
        dyn_action_dict['class_name'] = 'MyNewAction'
        dyn_action = dynamic_actions.DynamicAction(mock, dyn_action_dict)
        self.client.dynamic_actions.update.return_value = dyn_action
        result = self.call(dyn_actions_cmd.Update, app_args=['1-2-3-4', '--class-name', 'MyNewAction'])
        self.assertEqual(('1-2-3-4', 'my_action', 'MyNewAction', '2-3-4-5', 'my_module', '12345', 'private', '1', '1'), result[1])

    def test_list(self):
        self.client.dynamic_actions.list.return_value = [DYN_ACTION]
        result = self.call(dyn_actions_cmd.List)
        self.assertEqual([('1-2-3-4', 'my_action', 'MyAction', '2-3-4-5', 'my_module', '12345', 'private', '1', '1')], result[1])

    def test_get(self):
        self.client.dynamic_actions.get.return_value = DYN_ACTION
        result = self.call(dyn_actions_cmd.Get, app_args=['name'])
        self.assertEqual(('1-2-3-4', 'my_action', 'MyAction', '2-3-4-5', 'my_module', '12345', 'private', '1', '1'), result[1])

    def test_delete(self):
        self.call(dyn_actions_cmd.Delete, app_args=['my_action'])
        self.client.dynamic_actions.delete.assert_called_once_with('my_action', namespace='')

    def test_delete_with_multi_names(self):
        self.call(dyn_actions_cmd.Delete, app_args=['name1', 'name2'])
        self.assertEqual(2, self.client.dynamic_actions.delete.call_count)
        self.assertEqual([mock.call('name1', namespace=''), mock.call('name2', namespace='')], self.client.dynamic_actions.delete.call_args_list)