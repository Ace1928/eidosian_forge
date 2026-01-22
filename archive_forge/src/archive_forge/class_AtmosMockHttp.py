import sys
import base64
import os.path
import unittest
import libcloud.utils.files
from libcloud.test import MockHttp, make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.atmos import AtmosDriver, AtmosConnection
from libcloud.storage.drivers.dummy import DummyIterator
class AtmosMockHttp(MockHttp, unittest.TestCase):
    fixtures = StorageFileFixtures('atmos')
    upload_created = False
    upload_stream_created = False

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self)
        if kwargs.get('host', None) and kwargs.get('port', None):
            MockHttp.__init__(self, *args, **kwargs)
        self._upload_object_via_stream_first_request = True

    def runTest(self):
        pass

    def request(self, method, url, body=None, headers=None, raw=False, stream=False):
        headers = headers or {}
        parsed = urlparse.urlparse(url)
        if parsed.query.startswith('metadata/'):
            parsed = list(parsed)
            parsed[2] = parsed[2] + '/' + parsed[4]
            parsed[4] = ''
            url = urlparse.urlunparse(parsed)
        return super().request(method, url, body, headers, raw)

    def _rest_namespace_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('empty_directory_listing.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace_test_container_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('empty_directory_listing.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace_test_container(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace_test_container__metadata_system(self, method, url, body, headers):
        headers = {'x-emc-meta': 'objectid=b21cb59a2ba339d1afdd4810010b0a5aba2ab6b9'}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_20_26_20container__metadata_system(self, method, url, body, headers):
        headers = {'x-emc-meta': 'objectid=b21cb59a2ba339d1afdd4810010b0a5aba2ab6b9'}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_not_found__metadata_system(self, method, url, body, headers):
        body = self.fixtures.load('not_found.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _rest_namespace_test_create_container(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_test_create_container__metadata_system(self, method, url, body, headers):
        headers = {'x-emc-meta': 'objectid=31a27b593629a3fe59f887fd973fd953e80062ce'}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_create_container_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('already_exists.xml')
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _rest_namespace_foo_bar_container(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_foo_bar_container_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('not_found.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _rest_namespace_foo_bar_container_NOT_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('not_empty.xml')
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _rest_namespace_test_container_test_object_metadata_system(self, method, url, body, headers):
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_20_26_20container_test_20_26_20object_metadata_system(self, method, url, body, headers):
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_container_test_object_metadata_user(self, method, url, body, headers):
        meta = {'md5': '6b21c4a111ac178feacf9ec9d0c71f17', 'foo-bar': 'test 1', 'bar-foo': 'test 2'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_20_26_20container_test_20_26_20object_metadata_user(self, method, url, body, headers):
        meta = {'md5': '6b21c4a111ac178feacf9ec9d0c71f17', 'foo-bar': 'test 1', 'bar-foo': 'test 2'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_test_container_not_found_metadata_system(self, method, url, body, headers):
        body = self.fixtures.load('not_found.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _rest_namespace_foo_bar_container_foo_bar_object_DELETE(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_foo_20_26_20bar_container_foo_20_26_20bar_object_DELETE(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_foo_bar_container_foo_bar_object_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('not_found.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _rest_namespace_fbc_ftu_metadata_system(self, method, url, body, headers):
        if not self.upload_created:
            self.__class__.upload_created = True
            body = self.fixtures.load('not_found.xml')
            return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])
        self.__class__.upload_created = False
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftu_metadata_user(self, method, url, body, headers):
        self.assertTrue('x-emc-meta' in headers)
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsdn_metadata_system(self, method, url, body, headers):
        if not self.upload_stream_created:
            self.__class__.upload_stream_created = True
            body = self.fixtures.load('not_found.xml')
            return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])
        self.__class__.upload_stream_created = False
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsdn(self, method, url, body, headers):
        if self._upload_object_via_stream_first_request:
            self.assertTrue('Range' not in headers)
            self.assertEqual(method, 'POST')
            self._upload_object_via_stream_first_request = False
        else:
            self.assertTrue('Range' in headers)
            self.assertEqual(method, 'PUT')
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsdn_metadata_user(self, method, url, body, headers):
        self.assertTrue('x-emc-meta' in headers)
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsde_metadata_system(self, method, url, body, headers):
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsde(self, method, url, body, headers):
        if self._upload_object_via_stream_first_request:
            self.assertTrue('Range' not in headers)
            self._upload_object_via_stream_first_request = False
        else:
            self.assertTrue('Range' in headers)
        self.assertEqual(method, 'PUT')
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsde_metadata_user(self, method, url, body, headers):
        self.assertTrue('x-emc-meta' in headers)
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftsd_metadata_system(self, method, url, body, headers):
        meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
        headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
        return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _rest_namespace_foo_bar_container_foo_bar_object(self, method, url, body, headers):
        body = generate_random_data(1000)
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace_foo_20_26_20bar_container_foo_20_26_20bar_object(self, method, url, body, headers):
        body = generate_random_data(1000)
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_namespace_fbc_ftu(self, method, url, body, headers):
        return (httplib.CREATED, '', {}, httplib.responses[httplib.CREATED])