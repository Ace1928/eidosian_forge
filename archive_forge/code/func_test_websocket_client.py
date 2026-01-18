import unittest
from .ws_client import get_websocket_url
def test_websocket_client(self):
    for url, ws_url in [('http://localhost/api', 'ws://localhost/api'), ('https://localhost/api', 'wss://localhost/api'), ('https://domain.com/api', 'wss://domain.com/api'), ('https://api.domain.com/api', 'wss://api.domain.com/api'), ('http://api.domain.com', 'ws://api.domain.com'), ('https://api.domain.com', 'wss://api.domain.com'), ('http://api.domain.com/', 'ws://api.domain.com/'), ('https://api.domain.com/', 'wss://api.domain.com/')]:
        self.assertEqual(get_websocket_url(url), ws_url)