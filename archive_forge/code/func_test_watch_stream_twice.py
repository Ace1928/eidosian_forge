import unittest
from mock import Mock, call
from .watch import Watch
def test_watch_stream_twice(self):
    w = Watch(float)
    for step in ['first', 'second']:
        fake_resp = Mock()
        fake_resp.close = Mock()
        fake_resp.release_conn = Mock()
        fake_resp.read_chunked = Mock(return_value=['{"type": "ADDED", "object": 1}\n'] * 4)
        fake_api = Mock()
        fake_api.get_namespaces = Mock(return_value=fake_resp)
        fake_api.get_namespaces.__doc__ = ':return: V1NamespaceList'
        count = 1
        for e in w.stream(fake_api.get_namespaces):
            count += 1
            if count == 3:
                w.stop()
        self.assertEqual(count, 3)
        fake_api.get_namespaces.assert_called_once_with(_preload_content=False, watch=True)
        fake_resp.read_chunked.assert_called_once_with(decode_content=False)
        fake_resp.close.assert_called_once()
        fake_resp.release_conn.assert_called_once()