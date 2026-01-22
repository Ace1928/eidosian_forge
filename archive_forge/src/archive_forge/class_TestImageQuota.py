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
class TestImageQuota(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageQuota, self).setUp()

    def _get_image(self, location_count=1, image_size=10):
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = 'xyz'
        base_image.size = image_size
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        locations = []
        for i in range(location_count):
            locations.append({'url': 'file:///g/there/it/is%d' % i, 'metadata': {}, 'status': 'active'})
        image_values = {'id': 'xyz', 'owner': context.owner, 'status': 'active', 'size': image_size, 'locations': locations}
        db_api.image_create(context, image_values)
        return image

    def test_quota_allowed(self):
        quota = 10
        self.config(user_storage_quota=str(quota))
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = 'id'
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        data = '*' * quota
        base_image.set_data(data, size=None)
        image.set_data(data)
        self.assertEqual(quota, base_image.size)

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

    def test_quota_allowed_unit_b(self):
        self._test_quota_allowed_unit(10, '10B')

    def test_quota_allowed_unit_kb(self):
        self._test_quota_allowed_unit(10, '1KB')

    def test_quota_allowed_unit_mb(self):
        self._test_quota_allowed_unit(10, '1MB')

    def test_quota_allowed_unit_gb(self):
        self._test_quota_allowed_unit(10, '1GB')

    def test_quota_allowed_unit_tb(self):
        self._test_quota_allowed_unit(10, '1TB')

    def _quota_exceeded_size(self, quota, data, deleted=True, size=None):
        self.config(user_storage_quota=quota)
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = 'id'
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        if deleted:
            with patch.object(store_utils, 'safe_delete_from_backend'):
                store_utils.safe_delete_from_backend(context, image.image_id, base_image.locations[0])
        self.assertRaises(exception.StorageQuotaFull, image.set_data, data, size=size)

    def test_quota_exceeded_no_size(self):
        quota = 10
        data = '*' * (quota + 1)
        with patch.object(glance.api.common, 'get_remaining_quota', return_value=0):
            self._quota_exceeded_size(str(quota), data)

    def test_quota_exceeded_with_right_size(self):
        quota = 10
        data = '*' * (quota + 1)
        self._quota_exceeded_size(str(quota), data, size=len(data), deleted=False)

    def test_quota_exceeded_with_right_size_b(self):
        quota = 10
        data = '*' * (quota + 1)
        self._quota_exceeded_size('10B', data, size=len(data), deleted=False)

    def test_quota_exceeded_with_right_size_kb(self):
        quota = units.Ki
        data = '*' * (quota + 1)
        self._quota_exceeded_size('1KB', data, size=len(data), deleted=False)

    def test_quota_exceeded_with_lie_size(self):
        quota = 10
        data = '*' * (quota + 1)
        self._quota_exceeded_size(str(quota), data, deleted=False, size=quota - 1)

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

    def test_append_location(self):
        new_location = {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}
        image = self._get_image()
        pre_add_locations = image.locations[:]
        image.locations.append(new_location)
        pre_add_locations.append(new_location)
        self.assertEqual(image.locations, pre_add_locations)

    def test_insert_location(self):
        new_location = {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}
        image = self._get_image()
        pre_add_locations = image.locations[:]
        image.locations.insert(0, new_location)
        pre_add_locations.insert(0, new_location)
        self.assertEqual(image.locations, pre_add_locations)

    def test_extend_location(self):
        new_location = {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}
        image = self._get_image()
        pre_add_locations = image.locations[:]
        image.locations.extend([new_location])
        pre_add_locations.extend([new_location])
        self.assertEqual(image.locations, pre_add_locations)

    def test_iadd_location(self):
        new_location = {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}
        image = self._get_image()
        pre_add_locations = image.locations[:]
        image.locations += [new_location]
        pre_add_locations += [new_location]
        self.assertEqual(image.locations, pre_add_locations)

    def test_set_location(self):
        new_location = {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}
        image = self._get_image()
        image.locations = [new_location]
        self.assertEqual(image.locations, [new_location])

    def _make_image_with_quota(self, image_size=10, location_count=2):
        quota = image_size * location_count
        self.config(user_storage_quota=str(quota))
        return self._get_image(image_size=image_size, location_count=location_count)

    def test_exceed_append_location(self):
        image = self._make_image_with_quota()
        self.assertRaises(exception.StorageQuotaFull, image.locations.append, {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'})

    def test_exceed_insert_location(self):
        image = self._make_image_with_quota()
        self.assertRaises(exception.StorageQuotaFull, image.locations.insert, 0, {'url': 'file:///a/path', 'metadata': {}, 'status': 'active'})

    def test_exceed_extend_location(self):
        image = self._make_image_with_quota()
        self.assertRaises(exception.StorageQuotaFull, image.locations.extend, [{'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}])

    def test_set_location_under(self):
        image = self._make_image_with_quota(location_count=1)
        image.locations = [{'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}]

    def test_set_location_exceed(self):
        image = self._make_image_with_quota(location_count=1)
        try:
            image.locations = [{'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}, {'url': 'file:///a/path2', 'metadata': {}, 'status': 'active'}]
            self.fail('Should have raised the quota exception')
        except exception.StorageQuotaFull:
            pass

    def test_iadd_location_exceed(self):
        image = self._make_image_with_quota(location_count=1)
        try:
            image.locations += [{'url': 'file:///a/path', 'metadata': {}, 'status': 'active'}]
            self.fail('Should have raised the quota exception')
        except exception.StorageQuotaFull:
            pass

    def test_append_location_for_queued_image(self):
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = str(uuid.uuid4())
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        self.assertIsNone(image.size)
        self.mock_object(store_api, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        image.locations.append({'url': 'file:///fake.img.tar.gz', 'metadata': {}})
        self.assertIn({'url': 'file:///fake.img.tar.gz', 'metadata': {}}, image.locations)

    def test_insert_location_for_queued_image(self):
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = str(uuid.uuid4())
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        self.assertIsNone(image.size)
        self.mock_object(store_api, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        image.locations.insert(0, {'url': 'file:///fake.img.tar.gz', 'metadata': {}})
        self.assertIn({'url': 'file:///fake.img.tar.gz', 'metadata': {}}, image.locations)

    def test_set_location_for_queued_image(self):
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = str(uuid.uuid4())
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        self.assertIsNone(image.size)
        self.mock_object(store_api, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        image.locations = [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
        self.assertEqual([{'url': 'file:///fake.img.tar.gz', 'metadata': {}}], image.locations)

    def test_iadd_location_for_queued_image(self):
        context = FakeContext()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        base_image = FakeImage()
        base_image.image_id = str(uuid.uuid4())
        image = glance.quota.ImageProxy(base_image, context, db_api, store)
        self.assertIsNone(image.size)
        self.mock_object(store_api, 'get_size_from_backend', unit_test_utils.fake_get_size_from_backend)
        image.locations += [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
        self.assertIn({'url': 'file:///fake.img.tar.gz', 'metadata': {}}, image.locations)