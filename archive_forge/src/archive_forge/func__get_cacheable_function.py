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
def _get_cacheable_function(self, region=None):
    region = region if region else self.region
    memoize = cache.get_memoization_decorator(self.config_fixture.conf, region, group='cache')

    @memoize
    def cacheable_function(value=0, **kw):
        return value
    return cacheable_function