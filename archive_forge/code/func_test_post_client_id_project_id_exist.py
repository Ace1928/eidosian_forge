from unittest import mock
import uuid
from openstack.message.v2 import message
from openstack.tests.unit import base
def test_post_client_id_project_id_exist(self):
    sess = mock.Mock()
    resp = mock.Mock()
    sess.post.return_value = resp
    resources = ['/v2/queues/queue1/messages/578ee000508f153f256f717d/v2/queues/queue1/messages/579edd6c368cb61de9a7e233']
    resp.json.return_value = {'resources': resources}
    messages = [{'body': {'key': 'value1'}, 'ttl': 3600}, {'body': {'key': 'value2'}, 'ttl': 1800}]
    sot = message.Message(**FAKE2)
    res = sot.post(sess, messages)
    url = '/queues/%(queue)s/messages' % {'queue': FAKE2['queue_name']}
    headers = {'Client-ID': 'OLD_CLIENT_ID', 'X-PROJECT-ID': 'OLD_PROJECT_ID'}
    sess.post.assert_called_once_with(url, headers=headers, json={'messages': messages})
    resp.json.assert_called_once_with()
    self.assertEqual(resources, res)