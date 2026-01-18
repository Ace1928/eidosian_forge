import copy
from requests_mock.contrib import fixture
from openstackclient.object.v1 import container as container_cmds
from openstackclient.tests.unit.object.v1 import fakes as object_fakes
def test_object_show_container(self):
    headers = {'x-container-object-count': '42', 'x-container-bytes-used': '123', 'x-container-read': 'qaz', 'x-container-write': 'wsx', 'x-container-sync-to': 'edc', 'x-container-sync-key': 'rfv', 'x-storage-policy': 'o1--sr-r3'}
    self.requests_mock.register_uri('HEAD', object_fakes.ENDPOINT + '/ernie', headers=headers, status_code=200)
    arglist = ['ernie']
    verifylist = [('container', 'ernie')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = ('account', 'bytes_used', 'container', 'object_count', 'read_acl', 'storage_policy', 'sync_key', 'sync_to', 'write_acl')
    self.assertEqual(collist, columns)
    datalist = [object_fakes.ACCOUNT_ID, '123', 'ernie', '42', 'qaz', 'o1--sr-r3', 'rfv', 'edc', 'wsx']
    self.assertEqual(datalist, list(data))