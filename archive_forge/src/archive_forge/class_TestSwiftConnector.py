import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
@skipIf(missing_libs, skipmsg)
class TestSwiftConnector(TestCase):

    def setUp(self):
        super().setUp()
        self.conf = swift.load_conf(file=StringIO(config_file % def_config_file))
        with patch('geventhttpclient.HTTPClient.request', fake_auth_request_v1):
            self.conn = swift.SwiftConnector('fakerepo', conf=self.conf)

    def test_init_connector(self):
        self.assertEqual(self.conn.auth_ver, '1')
        self.assertEqual(self.conn.auth_url, 'http://127.0.0.1:8080/auth/v1.0')
        self.assertEqual(self.conn.user, 'test:tester')
        self.assertEqual(self.conn.password, 'testing')
        self.assertEqual(self.conn.root, 'fakerepo')
        self.assertEqual(self.conn.storage_url, 'http://127.0.0.1:8080/v1.0/AUTH_fakeuser')
        self.assertEqual(self.conn.token, '12' * 10)
        self.assertEqual(self.conn.http_timeout, 1)
        self.assertEqual(self.conn.http_pool_length, 1)
        self.assertEqual(self.conn.concurrency, 1)
        self.conf.set('swift', 'auth_ver', '2')
        self.conf.set('swift', 'auth_url', 'http://127.0.0.1:8080/auth/v2.0')
        with patch('geventhttpclient.HTTPClient.request', fake_auth_request_v2):
            conn = swift.SwiftConnector('fakerepo', conf=self.conf)
        self.assertEqual(conn.user, 'tester')
        self.assertEqual(conn.tenant, 'test')
        self.conf.set('swift', 'auth_ver', '1')
        self.conf.set('swift', 'auth_url', 'http://127.0.0.1:8080/auth/v1.0')
        with patch('geventhttpclient.HTTPClient.request', fake_auth_request_v1_error):
            self.assertRaises(swift.SwiftException, lambda: swift.SwiftConnector('fakerepo', conf=self.conf))

    def test_root_exists(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response()):
            self.assertEqual(self.conn.test_root_exists(), True)

    def test_root_not_exists(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(status=404)):
            self.assertEqual(self.conn.test_root_exists(), None)

    def test_create_root(self):
        with patch('dulwich.contrib.swift.SwiftConnector.test_root_exists', lambda *args: None):
            with patch('geventhttpclient.HTTPClient.request', lambda *args: Response()):
                self.assertEqual(self.conn.create_root(), None)

    def test_create_root_fails(self):
        with patch('dulwich.contrib.swift.SwiftConnector.test_root_exists', lambda *args: None):
            with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(status=404)):
                self.assertRaises(swift.SwiftException, self.conn.create_root)

    def test_get_container_objects(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(content=json.dumps(({'name': 'a'}, {'name': 'b'})))):
            self.assertEqual(len(self.conn.get_container_objects()), 2)

    def test_get_container_objects_fails(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(status=404)):
            self.assertEqual(self.conn.get_container_objects(), None)

    def test_get_object_stat(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(headers={'content-length': '10'})):
            self.assertEqual(self.conn.get_object_stat('a')['content-length'], '10')

    def test_get_object_stat_fails(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(status=404)):
            self.assertEqual(self.conn.get_object_stat('a'), None)

    def test_put_object(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response()):
            self.assertEqual(self.conn.put_object('a', BytesIO(b'content')), None)

    def test_put_object_fails(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(status=400)):
            self.assertRaises(swift.SwiftException, lambda: self.conn.put_object('a', BytesIO(b'content')))

    def test_get_object(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(content=b'content')):
            self.assertEqual(self.conn.get_object('a').read(), b'content')
        with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(content=b'content')):
            self.assertEqual(self.conn.get_object('a', range='0-6'), b'content')

    def test_get_object_fails(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(status=404)):
            self.assertEqual(self.conn.get_object('a'), None)

    def test_del_object(self):
        with patch('geventhttpclient.HTTPClient.request', lambda *args: Response()):
            self.assertEqual(self.conn.del_object('a'), None)

    def test_del_root(self):
        with patch('dulwich.contrib.swift.SwiftConnector.del_object', lambda *args: None):
            with patch('dulwich.contrib.swift.SwiftConnector.get_container_objects', lambda *args: ({'name': 'a'}, {'name': 'b'})):
                with patch('geventhttpclient.HTTPClient.request', lambda *args: Response()):
                    self.assertEqual(self.conn.del_root(), None)