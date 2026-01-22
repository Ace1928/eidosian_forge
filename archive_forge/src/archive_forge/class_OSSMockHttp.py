import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
class OSSMockHttp(MockHttp, unittest.TestCase):
    fixtures = StorageFileFixtures('oss')
    base_headers = {}

    def _unauthorized(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', self.base_headers, httplib.responses[httplib.OK])

    def _list_containers_empty(self, method, url, body, headers):
        body = self.fixtures.load('list_containers_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_containers(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_container_objects_empty(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_container_objects(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_container_objects_chinese(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects_chinese.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_container_objects_prefix(self, method, url, body, headers):
        params = {'prefix': self.test.prefix}
        self.assertUrlContainsQueryParams(url, params)
        body = self.fixtures.load('list_container_objects_prefix.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _get_container(self, method, url, body, headers):
        return self._list_containers(method, url, body, headers)

    def _get_object(self, method, url, body, headers):
        return self._list_containers(method, url, body, headers)

    def _notexisted_get_object(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.NOT_FOUND])

    def _test_get_object(self, method, url, body, headers):
        self.base_headers.update({'accept-ranges': 'bytes', 'connection': 'keep-alive', 'content-length': '0', 'content-type': 'application/octet-stream', 'date': 'Sat, 16 Jan 2016 15:38:14 GMT', 'etag': '"D41D8CD98F00B204E9800998ECF8427E"', 'last-modified': 'Fri, 15 Jan 2016 14:43:15 GMT', 'server': 'AliyunOSS', 'x-oss-object-type': 'Normal', 'x-oss-request-id': '569A63E6257784731E3D877F', 'x-oss-meta-rabbits': 'monkeys'})
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _invalid_name(self, method, url, body, headers):
        return (httplib.BAD_REQUEST, body, headers, httplib.responses[httplib.OK])

    def _already_exists(self, method, url, body, headers):
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.OK])

    def _create_container(self, method, url, body, headers):
        self.assertEqual('PUT', method)
        self.assertEqual('', body)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _create_container_location(self, method, url, body, headers):
        self.assertEqual('PUT', method)
        location_constraint = '<CreateBucketConfiguration><LocationConstraint>%s</LocationConstraint></CreateBucketConfiguration>' % self.test.ex_location
        self.assertEqual(location_constraint, body)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _delete_container_doesnt_exist(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.OK])

    def _delete_container_not_empty(self, method, url, body, headers):
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.OK])

    def _delete_container(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, self.base_headers, httplib.responses[httplib.NO_CONTENT])

    def _foo_bar_object_not_found(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_object_delete(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, headers, httplib.responses[httplib.OK])

    def _list_multipart(self, method, url, body, headers):
        query_string = urlparse.urlsplit(url).query
        query = parse_qs(query_string)
        if 'key-marker' not in query:
            body = self.fixtures.load('ex_iterate_multipart_uploads_p1.xml')
        else:
            body = self.fixtures.load('ex_iterate_multipart_uploads_p2.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_object(self, method, url, body, headers):
        body = generate_random_data(1000)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_object_invalid_size(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_test_stream_data_multipart(self, method, url, body, headers):
        headers = {}
        body = ''
        headers = {'etag': '"0cc175b9c0f1b6a831c399e269772661"'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])