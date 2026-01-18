import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_resume_update_env(self):
    self.client.executions.update.return_value = EXEC
    result = self.call(execution_cmd.Update, app_args=['id', '-s', 'RUNNING', '--env', '{"k1": "foobar"}'])
    self.assertEqual(EX_RESULT, result[1])