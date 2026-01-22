from unittest import mock
from openstack.clustering.v1 import profile_type
from openstack.tests.unit import base
class TestProfileType(base.TestCase):

    def test_basic(self):
        sot = profile_type.ProfileType()
        self.assertEqual('profile_type', sot.resource_key)
        self.assertEqual('profile_types', sot.resources_key)
        self.assertEqual('/profile-types', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)

    def test_instantiate(self):
        sot = profile_type.ProfileType(**FAKE)
        self.assertEqual(FAKE['name'], sot._get_id(sot))
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['schema'], sot.schema)
        self.assertEqual(FAKE['support_status'], sot.support_status)

    def test_ops(self):
        sot = profile_type.ProfileType(**FAKE)
        resp = mock.Mock()
        resp.json = mock.Mock(return_value='')
        sess = mock.Mock()
        sess.get = mock.Mock(return_value=resp)
        self.assertEqual('', sot.type_ops(sess))
        url = 'profile-types/%s/ops' % sot.id
        sess.get.assert_called_once_with(url)