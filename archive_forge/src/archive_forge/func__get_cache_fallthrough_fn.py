import copy
import ssl
import time
from unittest import mock
from dogpile.cache import proxy
from oslo_config import cfg
from oslo_utils import uuidutils
from pymemcache import KeepaliveOpts
from oslo_cache import _opts
from oslo_cache import core as cache
from oslo_cache import exception
from oslo_cache.tests import test_cache
def _get_cache_fallthrough_fn(self, cache_time):
    memoize = cache.get_memoization_decorator(self.config_fixture.conf, self.region, group='cache', expiration_group=TEST_GROUP2)

    class _test_obj(object):

        def __init__(self, value):
            self.test_value = value

        @memoize
        def get_test_value(self):
            return self.test_value

    def _do_test(value):
        test_obj = _test_obj(value)
        test_obj.get_test_value()
        cached_value = test_obj.get_test_value()
        self.assertTrue(cached_value.cached)
        self.assertEqual(value.value, cached_value.value)
        self.assertEqual(cached_value.value, test_obj.test_value.value)
        test_obj.test_value = TestProxyValue(uuidutils.generate_uuid(dashed=False))
        self.assertEqual(cached_value.value, test_obj.get_test_value().value)
        new_time = time.time() + cache_time * 2
        with mock.patch.object(time, 'time', return_value=new_time):
            overriden_cache_value = test_obj.get_test_value()
            self.assertNotEqual(cached_value.value, overriden_cache_value.value)
            self.assertEqual(test_obj.test_value.value, overriden_cache_value.value)
    return _do_test