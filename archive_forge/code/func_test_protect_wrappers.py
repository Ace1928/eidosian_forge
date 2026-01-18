import struct
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.tests.unit import utils
def test_protect_wrappers(self):
    data = b'My Pretty Little Data'
    for strategy in [b'MAC', b'ENCRYPT']:
        keys = self._setup_keys(strategy)
        protected = memcache_crypt.protect_data(keys, data)
        self.assertNotEqual(protected, data)
        if strategy == b'ENCRYPT':
            self.assertNotIn(data, protected)
        unprotected = memcache_crypt.unprotect_data(keys, protected)
        self.assertEqual(data, unprotected)
        self.assertRaises(memcache_crypt.InvalidMacError, memcache_crypt.unprotect_data, keys, protected[:-1])
        self.assertIsNone(memcache_crypt.unprotect_data(keys, None))