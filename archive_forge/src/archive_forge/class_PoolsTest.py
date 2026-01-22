from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.pools import Pool
class PoolsTest(utils.TestCase):

    def test_get_pool_stats(self):
        sl = cs.pools.list()
        cs.assert_called('GET', '/scheduler-stats/get_pools')
        self._assert_request_id(sl)
        for s in sl:
            self.assertIsInstance(s, Pool)
            self.assertTrue(hasattr(s, 'name'))
            self.assertFalse(hasattr(s, 'capabilities'))
            self.assertFalse(hasattr(s, 'volume_backend_name'))

    def test_get_detail_pool_stats(self):
        sl = cs.pools.list(detailed=True)
        self._assert_request_id(sl)
        cs.assert_called('GET', '/scheduler-stats/get_pools?detail=True')
        for s in sl:
            self.assertIsInstance(s, Pool)
            self.assertTrue(hasattr(s, 'name'))
            self.assertFalse(hasattr(s, 'capabilities'))
            self.assertTrue(hasattr(s, 'volume_backend_name'))