import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
@mock.patch.object(os.path, 'isfile', autospec=True)
@mock.patch.object(dogpile.cache.region, 'CacheRegion', autospec=True)
@mock.patch.object(filecache, '_get_cache', autospec=True)
def test_retrieve_data_ok(self, mock_get_cache, mock_cache, mock_isfile):
    s = 'spam'
    mock_isfile.return_value = True
    mock_cache.get.return_value = s
    mock_get_cache.return_value = mock_cache
    host = 'fred'
    port = '1234'
    hostport = '%s:%s' % (host, port)
    result = filecache.retrieve_data(host, port)
    mock_cache.get.assert_called_once_with(hostport, expiration_time=None)
    self.assertEqual(s, result)