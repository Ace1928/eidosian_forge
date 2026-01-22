import os
from oslo_cache.tests.functional import test_base
class TestBMemcachePoolCacheBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        MEMCACHED_PORT = os.getenv('OSLO_CACHE_TEST_MEMCACHED_PORT', '11211')
        self.config_fixture.config(group='cache', backend='oslo_cache.memcache_pool', enabled=True, memcache_servers=[f'localhost:{MEMCACHED_PORT}'], memcache_sasl_enabled=False, memcache_username='sasl_name', memcache_password='sasl_pswd')
        super(TestBMemcachePoolCacheBackend, self).setUp()