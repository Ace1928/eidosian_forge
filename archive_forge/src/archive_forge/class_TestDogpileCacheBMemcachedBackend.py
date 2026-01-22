from oslo_cache.tests.functional import test_base
class TestDogpileCacheBMemcachedBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        self.config_fixture.config(group='cache', backend='dogpile.cache.bmemcached', memcache_servers='localhost:11212')
        super().setUp()