import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_sub_executions_errors_only(self):
    self.client.executions.get_ex_sub_executions.return_value = EXECS_LIST
    self.call(execution_cmd.SubExecutionsLister, app_args=[EXEC_DICT['id'], '--errors-only'])
    self.assertEqual(1, self.client.executions.get_ex_sub_executions.call_count)
    self.assertEqual([mock.call(EXEC_DICT['id'], errors_only=True, max_depth=-1)], self.client.executions.get_ex_sub_executions.call_args_list)