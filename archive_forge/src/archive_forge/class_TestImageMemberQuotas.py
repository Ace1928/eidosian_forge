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
class TestImageMemberQuotas(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageMemberQuotas, self).setUp()
        db_api = unit_test_utils.FakeDB()
        store_api = unit_test_utils.FakeStoreAPI()
        store = unit_test_utils.FakeStoreUtils(store_api)
        context = FakeContext()
        self.image = mock.Mock()
        self.base_image_member_factory = mock.Mock()
        self.image_member_factory = glance.quota.ImageMemberFactoryProxy(self.base_image_member_factory, context, db_api, store)

    def test_new_image_member(self):
        self.config(image_member_quota=1)
        self.image_member_factory.new_image_member(self.image, 'fake_id')
        nim = self.base_image_member_factory.new_image_member
        nim.assert_called_once_with(self.image, 'fake_id')

    def test_new_image_member_unlimited_members(self):
        self.config(image_member_quota=-1)
        self.image_member_factory.new_image_member(self.image, 'fake_id')
        nim = self.base_image_member_factory.new_image_member
        nim.assert_called_once_with(self.image, 'fake_id')

    def test_new_image_member_too_many_members(self):
        self.config(image_member_quota=0)
        self.assertRaises(exception.ImageMemberLimitExceeded, self.image_member_factory.new_image_member, self.image, 'fake_id')