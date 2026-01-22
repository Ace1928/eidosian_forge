from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import type
from openstack.tests.unit import base
class TestType(base.TestCase):

    def setUp(self):
        super(TestType, self).setUp()
        self.extra_specs_result = {'extra_specs': {'go': 'cubs', 'boo': 'sox'}}
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.status_code = 200
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = '3.0'
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_basic(self):
        sot = type.Type(**TYPE)
        self.assertEqual('volume_type', sot.resource_key)
        self.assertEqual('volume_types', sot.resources_key)
        self.assertEqual('/types', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_commit)

    def test_new(self):
        sot = type.Type.new(id=FAKE_ID)
        self.assertEqual(FAKE_ID, sot.id)

    def test_create(self):
        sot = type.Type(**TYPE)
        self.assertEqual(TYPE['id'], sot.id)
        self.assertEqual(TYPE['extra_specs'], sot.extra_specs)
        self.assertEqual(TYPE['name'], sot.name)

    def test_get_private_access(self):
        sot = type.Type(**TYPE)
        response = mock.Mock()
        response.status_code = 200
        response.body = {'volume_type_access': [{'project_id': 'a', 'volume_type_id': 'b'}]}
        response.json = mock.Mock(return_value=response.body)
        self.sess.get = mock.Mock(return_value=response)
        self.assertEqual(response.body['volume_type_access'], sot.get_private_access(self.sess))
        self.sess.get.assert_called_with('types/%s/os-volume-type-access' % sot.id)

    def test_add_private_access(self):
        sot = type.Type(**TYPE)
        self.assertIsNone(sot.add_private_access(self.sess, 'a'))
        url = 'types/%s/action' % sot.id
        body = {'addProjectAccess': {'project': 'a'}}
        self.sess.post.assert_called_with(url, json=body)

    def test_remove_private_access(self):
        sot = type.Type(**TYPE)
        self.assertIsNone(sot.remove_private_access(self.sess, 'a'))
        url = 'types/%s/action' % sot.id
        body = {'removeProjectAccess': {'project': 'a'}}
        self.sess.post.assert_called_with(url, json=body)