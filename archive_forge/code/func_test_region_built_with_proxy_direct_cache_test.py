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
def test_region_built_with_proxy_direct_cache_test(self):
    test_value = TestProxyValue('Direct Cache Test')
    self.region.set('cache_test', test_value)
    cached_value = self.region.get('cache_test')
    self.assertTrue(cached_value.cached)