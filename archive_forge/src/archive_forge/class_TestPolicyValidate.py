from openstack.clustering.v1 import policy
from openstack.tests.unit import base
class TestPolicyValidate(base.TestCase):

    def setUp(self):
        super(TestPolicyValidate, self).setUp()

    def test_basic(self):
        sot = policy.PolicyValidate()
        self.assertEqual('policy', sot.resource_key)
        self.assertEqual('policies', sot.resources_key)
        self.assertEqual('/policies/validate', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)