from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
class TestCacheImageAPIPolicy(utils.BaseTestCase):

    def setUp(self):
        super(TestCacheImageAPIPolicy, self).setUp()
        self.enforcer = mock.MagicMock()
        self.context = mock.MagicMock()

    def test_manage_image_cache(self):
        self.policy = policy.CacheImageAPIPolicy(self.context, enforcer=self.enforcer, policy_str='manage_image_cache')
        self.policy.manage_image_cache()
        self.enforcer.enforce.assert_called_once_with(self.context, 'manage_image_cache', mock.ANY)

    def test_manage_image_cache_with_cache_delete(self):
        self.policy = policy.CacheImageAPIPolicy(self.context, enforcer=self.enforcer, policy_str='cache_delete')
        self.policy.manage_image_cache()
        self.enforcer.enforce.assert_called_once_with(self.context, 'cache_delete', mock.ANY)

    def test_manage_image_cache_with_cache_list(self):
        self.policy = policy.CacheImageAPIPolicy(self.context, enforcer=self.enforcer, policy_str='cache_list')
        self.policy.manage_image_cache()
        self.enforcer.enforce.assert_called_once_with(self.context, 'cache_list', mock.ANY)

    def test_manage_image_cache_with_cache_image(self):
        self.policy = policy.CacheImageAPIPolicy(self.context, enforcer=self.enforcer, policy_str='cache_image')
        self.policy.manage_image_cache()
        self.enforcer.enforce.assert_called_once_with(self.context, 'cache_image', mock.ANY)