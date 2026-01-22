import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1_1FlavorUnitTest(base.QueuesTestBase):

    def test_flavor_create(self):
        pool_list = ['pool1', 'pool2']
        flavor_data = {'pool_list': pool_list}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            flavor = self.client.flavor('tasty', **flavor_data)
            self.assertEqual('tasty', flavor.name)

    def test_flavor_get(self):
        flavor_data = {'name': 'test'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(flavor_data))
            send_method.return_value = resp
            flavor = self.client.flavor('test')
            flavor1 = flavor.get()
            self.assertEqual('test', flavor1['name'])

    def test_flavor_update(self):
        pool_list1 = ['pool1', 'pool2']
        pool_list2 = ['pool3', 'pool4']
        flavor_data = {'pool_list': pool_list1}
        updated_data = {'pool_list': pool_list2}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(updated_data))
            send_method.return_value = resp
            flavor = self.client.flavor('tasty', **flavor_data)
            flavor.update({'pool_list': pool_list2})
            self.assertEqual(pool_list2, flavor.pool_list)

    def test_flavor_list(self):
        returned = {'links': [{'rel': 'next', 'href': '/v1.1/flavors?marker=6244-244224-783'}], 'flavors': [{'name': 'tasty'}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            flavor_var = self.client.flavors(limit=1)
            self.assertIsInstance(flavor_var, iterator._Iterator)
            self.assertEqual(1, len(list(flavor_var)))

    def test_flavor_delete(self):
        pool_list = ['pool1', 'pool2']
        flavor_data = {'pool_list': pool_list}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            resp_data = response.Response(None, json.dumps(flavor_data))
            send_method.side_effect = iter([resp_data, resp])
            flavor = self.client.flavor('tasty', **flavor_data)
            flavor.delete()