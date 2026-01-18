import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_quota_exceeded_keystone_quotas(self):
    self.config(user_storage_quota='10B')
    context = FakeContext()
    db_api = unit_test_utils.FakeDB()
    store_api = unit_test_utils.FakeStoreAPI()
    store = unit_test_utils.FakeStoreUtils(store_api)
    base_image = FakeImage()
    base_image.image_id = 'id'
    image = glance.quota.ImageProxy(base_image, context, db_api, store)
    data = '*' * 100
    self.assertRaises(exception.StorageQuotaFull, image.set_data, data, size=len(data))
    self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
    self.config(use_keystone_limits=True)
    image.set_data(data, size=len(data))