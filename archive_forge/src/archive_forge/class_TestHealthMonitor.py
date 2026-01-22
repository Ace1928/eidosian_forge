from openstack.network.v2 import health_monitor
from openstack.tests.unit import base
class TestHealthMonitor(base.TestCase):

    def test_basic(self):
        sot = health_monitor.HealthMonitor()
        self.assertEqual('healthmonitor', sot.resource_key)
        self.assertEqual('healthmonitors', sot.resources_key)
        self.assertEqual('/lbaas/healthmonitors', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = health_monitor.HealthMonitor(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['delay'], sot.delay)
        self.assertEqual(EXAMPLE['expected_codes'], sot.expected_codes)
        self.assertEqual(EXAMPLE['http_method'], sot.http_method)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['max_retries'], sot.max_retries)
        self.assertEqual(EXAMPLE['pools'], sot.pool_ids)
        self.assertEqual(EXAMPLE['pool_id'], sot.pool_id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['timeout'], sot.timeout)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['url_path'], sot.url_path)
        self.assertEqual(EXAMPLE['name'], sot.name)