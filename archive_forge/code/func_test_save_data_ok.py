import os
from unittest import mock
import dogpile.cache
from ironicclient.common import filecache
from ironicclient.tests.unit import utils
@mock.patch.object(dogpile.cache.region, 'CacheRegion', autospec=True)
@mock.patch.object(filecache, '_get_cache', autospec=True)
def test_save_data_ok(self, mock_get_cache, mock_cache):
    mock_get_cache.return_value = mock_cache
    host = 'fred'
    port = '1234'
    hostport = '%s:%s' % (host, port)
    data = 'some random data'
    filecache.save_data(host, port, data)
    mock_cache.set.assert_called_once_with(hostport, data)