import abc
from keystoneauth1 import identity
from keystoneauth1 import session
from oslo_config import cfg
from oslo_context import context
from oslo_utils import uuidutils
from oslotest import base
from testtools import testcase
from castellan.common.credentials import keystone_password
from castellan.common.credentials import keystone_token
from castellan.common import exception
from castellan.key_manager import barbican_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
from castellan.tests import utils
def test_secret_create_check_consumers_list_consistency(self):
    """Consumers List Consistency

        Check that the consumers list contains a single element,
        and that it corresponds to the consumer created.
        """
    key = test_key_manager._get_test_passphrase()
    self.assertIsNotNone(key)
    stored_id = self.key_mgr.store(self.ctxt, key)
    self.addCleanup(self.key_mgr.delete, self.ctxt, stored_id, True)
    self.assertIsNotNone(stored_id)
    resource_id = uuidutils.generate_uuid()
    consumer_data = {'service': 'dummy_service', 'resource_type': 'dummy_resource_type', 'resource_id': resource_id}
    self.key_mgr.add_consumer(self.ctxt, stored_id, consumer_data)
    stored_secret = self.key_mgr.get(self.ctxt, stored_id)
    self.assertIsNotNone(stored_secret)
    self.assertIsInstance(stored_secret.consumers, list)
    self.assertEqual(len(stored_secret.consumers), 1)
    self.assertEqual(stored_secret.consumers[0]['service'], consumer_data['service'])
    self.assertEqual(stored_secret.consumers[0]['resource_type'], consumer_data['resource_type'])
    self.assertEqual(stored_secret.consumers[0]['resource_id'], consumer_data['resource_id'])