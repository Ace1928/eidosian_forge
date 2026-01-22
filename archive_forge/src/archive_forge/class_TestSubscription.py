import copy
from unittest import mock
import uuid
from openstack.message.v2 import subscription
from openstack.tests.unit import base
class TestSubscription(base.TestCase):

    def test_basic(self):
        sot = subscription.Subscription()
        self.assertEqual('subscriptions', sot.resources_key)
        self.assertEqual('/queues/%(queue_name)s/subscriptions', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = subscription.Subscription.new(**FAKE2)
        self.assertEqual(FAKE2['age'], sot.age)
        self.assertEqual(FAKE2['id'], sot.id)
        self.assertEqual(FAKE2['options'], sot.options)
        self.assertEqual(FAKE2['source'], sot.source)
        self.assertEqual(FAKE2['subscriber'], sot.subscriber)
        self.assertEqual(FAKE2['subscription_id'], sot.subscription_id)
        self.assertEqual(FAKE2['ttl'], sot.ttl)
        self.assertEqual(FAKE2['queue_name'], sot.queue_name)
        self.assertEqual(FAKE2['client_id'], sot.client_id)
        self.assertEqual(FAKE2['project_id'], sot.project_id)

    @mock.patch.object(uuid, 'uuid4')
    def test_create(self, mock_uuid):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.post.return_value = resp
        sess.get_project_id.return_value = 'NEW_PROJECT_ID'
        mock_uuid.return_value = 'NEW_CLIENT_ID'
        FAKE = copy.deepcopy(FAKE1)
        sot = subscription.Subscription(**FAKE1)
        sot._translate_response = mock.Mock()
        res = sot.create(sess)
        url = '/queues/%(queue)s/subscriptions' % {'queue': FAKE.pop('queue_name')}
        headers = {'Client-ID': 'NEW_CLIENT_ID', 'X-PROJECT-ID': 'NEW_PROJECT_ID'}
        sess.post.assert_called_once_with(url, headers=headers, json=FAKE)
        sess.get_project_id.assert_called_once_with()
        self.assertEqual(sot, res)

    def test_create_client_id_project_id_exist(self):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.post.return_value = resp
        FAKE = copy.deepcopy(FAKE2)
        sot = subscription.Subscription(**FAKE2)
        sot._translate_response = mock.Mock()
        res = sot.create(sess)
        url = '/queues/%(queue)s/subscriptions' % {'queue': FAKE.pop('queue_name')}
        headers = {'Client-ID': FAKE.pop('client_id'), 'X-PROJECT-ID': FAKE.pop('project_id')}
        sess.post.assert_called_once_with(url, headers=headers, json=FAKE)
        self.assertEqual(sot, res)

    @mock.patch.object(uuid, 'uuid4')
    def test_get(self, mock_uuid):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.get.return_value = resp
        sess.get_project_id.return_value = 'NEW_PROJECT_ID'
        mock_uuid.return_value = 'NEW_CLIENT_ID'
        sot = subscription.Subscription(**FAKE1)
        sot._translate_response = mock.Mock()
        res = sot.fetch(sess)
        url = 'queues/%(queue)s/subscriptions/%(subscription)s' % {'queue': FAKE1['queue_name'], 'subscription': FAKE1['id']}
        headers = {'Client-ID': 'NEW_CLIENT_ID', 'X-PROJECT-ID': 'NEW_PROJECT_ID'}
        sess.get.assert_called_with(url, headers=headers, skip_cache=False)
        sess.get_project_id.assert_called_once_with()
        sot._translate_response.assert_called_once_with(resp)
        self.assertEqual(sot, res)

    def test_get_client_id_project_id_exist(self):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.get.return_value = resp
        sot = subscription.Subscription(**FAKE2)
        sot._translate_response = mock.Mock()
        res = sot.fetch(sess)
        url = 'queues/%(queue)s/subscriptions/%(subscription)s' % {'queue': FAKE2['queue_name'], 'subscription': FAKE2['id']}
        headers = {'Client-ID': 'OLD_CLIENT_ID', 'X-PROJECT-ID': 'OLD_PROJECT_ID'}
        sess.get.assert_called_with(url, headers=headers, skip_cache=False)
        sot._translate_response.assert_called_once_with(resp)
        self.assertEqual(sot, res)

    @mock.patch.object(uuid, 'uuid4')
    def test_delete(self, mock_uuid):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.delete.return_value = resp
        sess.get_project_id.return_value = 'NEW_PROJECT_ID'
        mock_uuid.return_value = 'NEW_CLIENT_ID'
        sot = subscription.Subscription(**FAKE1)
        sot._translate_response = mock.Mock()
        sot.delete(sess)
        url = 'queues/%(queue)s/subscriptions/%(subscription)s' % {'queue': FAKE1['queue_name'], 'subscription': FAKE1['id']}
        headers = {'Client-ID': 'NEW_CLIENT_ID', 'X-PROJECT-ID': 'NEW_PROJECT_ID'}
        sess.delete.assert_called_with(url, headers=headers)
        sess.get_project_id.assert_called_once_with()
        sot._translate_response.assert_called_once_with(resp, has_body=False)

    def test_delete_client_id_project_id_exist(self):
        sess = mock.Mock()
        resp = mock.Mock()
        sess.delete.return_value = resp
        sot = subscription.Subscription(**FAKE2)
        sot._translate_response = mock.Mock()
        sot.delete(sess)
        url = 'queues/%(queue)s/subscriptions/%(subscription)s' % {'queue': FAKE2['queue_name'], 'subscription': FAKE2['id']}
        headers = {'Client-ID': 'OLD_CLIENT_ID', 'X-PROJECT-ID': 'OLD_PROJECT_ID'}
        sess.delete.assert_called_with(url, headers=headers)
        sot._translate_response.assert_called_once_with(resp, has_body=False)