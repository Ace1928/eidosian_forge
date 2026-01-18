from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
@mock.patch('oslo_utils.timeutils.StopWatch')
def test_proxy_read(self, mock_sw):
    items = ['1', '2', '3']
    source = mock.MagicMock()
    source.read.side_effect = items
    callback = mock.MagicMock()
    mock_sw.return_value.expired.side_effect = [False, True, False]
    cb_iter = script_utils.CallbackIterator(source, callback, min_interval=30)
    results = [cb_iter.read(1) for i in range(len(items))]
    self.assertEqual(items, results)
    callback.assert_has_calls([mock.call(2, 2)])
    cb_iter.close()
    callback.assert_has_calls([mock.call(2, 2), mock.call(1, 3)])