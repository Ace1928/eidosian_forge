import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_create_container_single(self):
    self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159'}, status_code=200)
    arglist = ['ernie']
    verifylist = [('containers', ['ernie'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159')]
    self.assertEqual(datalist, list(data))