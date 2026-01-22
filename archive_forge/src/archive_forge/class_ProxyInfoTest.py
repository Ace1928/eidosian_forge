import os
import unittest
from websocket._url import (
class ProxyInfoTest(unittest.TestCase):

    def setUp(self):
        self.http_proxy = os.environ.get('http_proxy', None)
        self.https_proxy = os.environ.get('https_proxy', None)
        self.no_proxy = os.environ.get('no_proxy', None)
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        if 'no_proxy' in os.environ:
            del os.environ['no_proxy']

    def tearDown(self):
        if self.http_proxy:
            os.environ['http_proxy'] = self.http_proxy
        elif 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if self.https_proxy:
            os.environ['https_proxy'] = self.https_proxy
        elif 'https_proxy' in os.environ:
            del os.environ['https_proxy']
        if self.no_proxy:
            os.environ['no_proxy'] = self.no_proxy
        elif 'no_proxy' in os.environ:
            del os.environ['no_proxy']

    def testProxyFromArgs(self):
        self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost'), ('localhost', 0, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_port=3128), ('localhost', 3128, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost'), ('localhost', 0, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128), ('localhost', 3128, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_auth=('a', 'b')), ('localhost', 0, ('a', 'b')))
        self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_port=3128, proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_auth=('a', 'b')), ('localhost', 0, ('a', 'b')))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, no_proxy=['example.com'], proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
        self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, no_proxy=['echo.websocket.events'], proxy_auth=('a', 'b')), (None, 0, None))

    def testProxyFromEnv(self):
        os.environ['http_proxy'] = 'http://localhost/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
        os.environ['http_proxy'] = 'http://localhost:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
        os.environ['http_proxy'] = 'http://localhost/'
        os.environ['https_proxy'] = 'http://localhost2/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
        os.environ['http_proxy'] = 'http://localhost:3128/'
        os.environ['https_proxy'] = 'http://localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
        os.environ['http_proxy'] = 'http://localhost/'
        os.environ['https_proxy'] = 'http://localhost2/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, None))
        os.environ['http_proxy'] = 'http://localhost:3128/'
        os.environ['https_proxy'] = 'http://localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, None))
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = 'http://localhost2/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False), (None, 0, None))
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = 'http://localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False), (None, 0, None))
        os.environ['http_proxy'] = 'http://localhost/'
        os.environ['https_proxy'] = ''
        self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, None))
        os.environ['http_proxy'] = 'http://localhost:3128/'
        os.environ['https_proxy'] = ''
        self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, None))
        os.environ['http_proxy'] = 'http://a:b@localhost/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost/'
        os.environ['https_proxy'] = 'http://a:b@localhost2/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', None, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', False), ('localhost', 3128, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost/'
        os.environ['https_proxy'] = 'http://a:b@localhost2/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', None, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, ('a', 'b')))
        os.environ['http_proxy'] = 'http://john%40example.com:P%40SSWORD@localhost:3128/'
        os.environ['https_proxy'] = 'http://john%40example.com:P%40SSWORD@localhost2:3128/'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), ('localhost2', 3128, ('john@example.com', 'P@SSWORD')))
        os.environ['http_proxy'] = 'http://a:b@localhost/'
        os.environ['https_proxy'] = 'http://a:b@localhost2/'
        os.environ['no_proxy'] = 'example1.com,example2.com'
        self.assertEqual(get_proxy_info('example.1.com', True), ('localhost2', None, ('a', 'b')))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
        os.environ['no_proxy'] = 'example1.com,example2.com, echo.websocket.events'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
        os.environ['no_proxy'] = 'example1.com,example2.com, .websocket.events'
        self.assertEqual(get_proxy_info('echo.websocket.events', True), (None, 0, None))
        os.environ['http_proxy'] = 'http://a:b@localhost:3128/'
        os.environ['https_proxy'] = 'http://a:b@localhost2:3128/'
        os.environ['no_proxy'] = '127.0.0.0/8, 192.168.0.0/16'
        self.assertEqual(get_proxy_info('127.0.0.1', False), (None, 0, None))
        self.assertEqual(get_proxy_info('192.168.1.1', False), (None, 0, None))