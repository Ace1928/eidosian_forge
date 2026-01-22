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
def _test_quota_allowed_unit(self, data_length, config_quota):
    self.config(user_storage_quota=config_quota)
    context = FakeContext()
    db_api = unit_test_utils.FakeDB()
    store_api = unit_test_utils.FakeStoreAPI()
    store = unit_test_utils.FakeStoreUtils(store_api)
    base_image = FakeImage()
    base_image.image_id = 'id'
    image = glance.quota.ImageProxy(base_image, context, db_api, store)
    data = '*' * data_length
    base_image.set_data(data, size=None)
    image.set_data(data)
    self.assertEqual(data_length, base_image.size)