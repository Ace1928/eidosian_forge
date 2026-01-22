import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
class QueuesV2SubscriptionUnitTest(base.QueuesTestBase):

    def test_subscription_create(self):
        subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            create_resp = response.Response(None, '{"subscription_id": "fake_id"}')
            get_content = '{"subscriber": "http://trigger.me","ttl": 3600, "id": "fake_id"}'
            get_resp = response.Response(None, get_content)
            send_method.side_effect = iter([create_resp, get_resp])
            subscription = self.client.subscription('beijing', **subscription_data)
            self.assertEqual('http://trigger.me', subscription.subscriber)
            self.assertEqual(3600, subscription.ttl)
            self.assertEqual('fake_id', subscription.id)

    def test_subscription_create_duplicate_throws_conflicterror(self):
        subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        with mock.patch.object(self.transport.client, 'request', autospec=True) as request_method:

            class FakeRawResponse(object):

                def __init__(self):
                    self.text = ''
                    self.headers = {}
                    self.status_code = 409
            request_method.return_value = FakeRawResponse()
            self.assertRaises(errors.ConflictError, self.client.subscription, 'beijing', **subscription_data)

    def test_subscription_update(self):
        subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            create_resp = response.Response(None, '{"subscription_id": "fake_id"}')
            get_content = '{"subscriber": "http://trigger.me","ttl": 3600, "id": "fake_id"}'
            get_resp = response.Response(None, get_content)
            update_content = json.dumps({'subscriber': 'fake_subscriber'})
            update_resp = response.Response(None, update_content)
            send_method.side_effect = iter([create_resp, get_resp, update_resp])
            subscription = self.client.subscription('beijing', **subscription_data)
            subscription.update({'subscriber': 'fake_subscriber'})
            self.assertEqual('fake_subscriber', subscription.subscriber)

    def test_subscription_delete(self):
        subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            create_resp = response.Response(None, '{"subscription_id": "fake_id"}')
            get_content = '{"subscriber": "http://trigger.me","ttl": 3600, "id": "fake_id"}'
            get_resp = response.Response(None, get_content)
            send_method.side_effect = iter([create_resp, get_resp, None, errors.ResourceNotFound])
            subscription = self.client.subscription('beijing', **subscription_data)
            self.assertEqual('http://trigger.me', subscription.subscriber)
            self.assertEqual(3600, subscription.ttl)
            self.assertEqual('fake_id', subscription.id)
            subscription.delete()
            self.assertRaises(errors.ResourceNotFound, self.client.subscription, 'beijing', **{'id': 'fake_id'})

    def test_subscription_get(self):
        subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(subscription_data))
            send_method.return_value = resp
            kwargs = {'id': 'fake_id'}
            subscription = self.client.subscription('test', **kwargs)
            self.assertEqual('http://trigger.me', subscription.subscriber)
            self.assertEqual(3600, subscription.ttl)

    def test_subscription_list(self):
        subscription_data = {'subscriptions': [{'source': 'beijing', 'id': '568afabb508f153573f6a56f', 'subscriber': 'http://trigger.me', 'ttl': 3600, 'age': 1800, 'confirmed': False, 'options': {}}, {'source': 'beijing', 'id': '568afabb508f153573f6a56x', 'subscriber': 'http://trigger.you', 'ttl': 7200, 'age': 2309, 'confirmed': True, 'options': {'oh stop': 'triggering'}}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            list_resp = response.Response(None, json.dumps(subscription_data))
            send_method.side_effect = iter([list_resp])
            subscriptions = list(self.client.subscriptions('beijing'))
            self.assertEqual(2, len(subscriptions))
            subscriber_list = [s.subscriber for s in subscriptions]
            self.assertIn('http://trigger.me', subscriber_list)
            self.assertIn('http://trigger.you', subscriber_list)
            for sub in subscriptions:
                if sub.subscriber == 'http://trigger.you':
                    self.assertEqual('beijing', sub.queue_name)
                    self.assertEqual('568afabb508f153573f6a56x', sub.id)
                    self.assertEqual(7200, sub.ttl)
                    self.assertEqual({'oh stop': 'triggering'}, sub.options)