import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_update_invalid_state(self):
    states = ['IDLE', 'WAITING', 'DELAYED']
    for state in states:
        self.assertRaises(SystemExit, self.call, execution_cmd.Update, app_args=['id', '-s', state])