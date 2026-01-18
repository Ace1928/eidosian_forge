import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_create_container_storage_policy(self):
    self.requests_mock.register_uri('PUT', object_fakes.ENDPOINT + '/ernie', headers={'x-trans-id': '314159', 'x-storage-policy': 'o1--sr-r3'}, status_code=200)
    arglist = ['ernie', '--storage-policy', 'o1--sr-r3']
    verifylist = [('containers', ['ernie']), ('storage_policy', 'o1--sr-r3')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = [(object_fakes.ACCOUNT_ID, 'ernie', '314159')]
    self.assertEqual(datalist, list(data))