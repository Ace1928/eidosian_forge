from openstack.clustering.v1 import action
from openstack.tests.unit import base
class TestAction(base.TestCase):

    def setUp(self):
        super(TestAction, self).setUp()

    def test_basic(self):
        sot = action.Action()
        self.assertEqual('action', sot.resource_key)
        self.assertEqual('actions', sot.resources_key)
        self.assertEqual('/actions', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_commit)

    def test_instantiate(self):
        sot = action.Action(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['target'], sot.target_id)
        self.assertEqual(FAKE['action'], sot.action)
        self.assertEqual(FAKE['cause'], sot.cause)
        self.assertEqual(FAKE['owner'], sot.owner_id)
        self.assertEqual(FAKE['user'], sot.user_id)
        self.assertEqual(FAKE['project'], sot.project_id)
        self.assertEqual(FAKE['domain'], sot.domain_id)
        self.assertEqual(FAKE['interval'], sot.interval)
        self.assertEqual(FAKE['start_time'], sot.start_at)
        self.assertEqual(FAKE['end_time'], sot.end_at)
        self.assertEqual(FAKE['timeout'], sot.timeout)
        self.assertEqual(FAKE['status'], sot.status)
        self.assertEqual(FAKE['status_reason'], sot.status_reason)
        self.assertEqual(FAKE['inputs'], sot.inputs)
        self.assertEqual(FAKE['outputs'], sot.outputs)
        self.assertEqual(FAKE['depends_on'], sot.depends_on)
        self.assertEqual(FAKE['depended_by'], sot.depended_by)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)
        self.assertEqual(FAKE['cluster_id'], sot.cluster_id)