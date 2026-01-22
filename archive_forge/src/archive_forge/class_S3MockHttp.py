import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
class S3MockHttp(BaseRangeDownloadMockHttp, unittest.TestCase):
    fixtures = StorageFileFixtures('s3')
    base_headers = {}

    def _UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', self.base_headers, httplib.responses[httplib.OK])

    def _DIFFERENT_REGION(self, method, url, body, headers):
        return (httplib.MOVED_PERMANENTLY, '', self.base_headers, httplib.responses[httplib.OK])

    def _list_containers_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_containers_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_containers_TOKEN(self, method, url, body, headers):
        if 'x-amz-security-token' in headers:
            assert headers['x-amz-security-token'] == 'asdf'
        body = self.fixtures.load('list_containers_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _list_containers(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test_container_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects_empty.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test_container(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test_container_ITERATOR(self, method, url, body, headers):
        if url.find('3.zip') == -1:
            file_name = 'list_container_objects_not_exhausted1.xml'
        else:
            file_name = 'list_container_objects_not_exhausted2.xml'
        body = self.fixtures.load(file_name)
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test2_get_object(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _test2_test_get_object_no_content_type(self, method, url, body, headers):
        headers = {'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test2_get_object_no_content_type(self, method, url, body, headers):
        headers = {'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test2_test_get_object(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test2_get_object_no_content_length(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _test2_test_get_object_no_content_length(self, method, url, body, headers):
        body = self.fixtures.load('list_containers.xml')
        headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _new_container_INVALID_NAME(self, method, url, body, headers):
        return (httplib.BAD_REQUEST, body, headers, httplib.responses[httplib.OK])

    def _new_container_ALREADY_EXISTS(self, method, url, body, headers):
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.OK])

    def _new_container(self, method, url, body, headers):
        if method == 'PUT':
            status = httplib.OK
        elif method == 'DELETE':
            status = httplib.NO_CONTENT
        return (status, body, headers, httplib.responses[httplib.OK])

    def _new_container_DOESNT_EXIST(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.OK])

    def _new_container_NOT_EMPTY(self, method, url, body, headers):
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.OK])

    def _test1_get_container(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects.xml')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _container1_get_container(self, method, url, body, headers):
        return (httplib.NOT_FOUND, '', self.base_headers, httplib.responses[httplib.NOT_FOUND])

    def _test_inexistent_get_object(self, method, url, body, headers):
        return (httplib.NOT_FOUND, '', self.base_headers, httplib.responses[httplib.NOT_FOUND])

    def _foo_bar_container(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_NOT_FOUND(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_bar_object_NOT_FOUND(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_bar_object_DELETE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_test_stream_data(self, method, url, body, headers):
        body = ''
        headers = {'etag': '"0cc175b9c0f1b6a831c399e269772661"'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_test_stream_data_MULTIPART(self, method, url, body, headers):
        if method == 'POST':
            if 'uploadId' in url:
                body = self.fixtures.load('complete_multipart.xml')
                return (httplib.OK, body, headers, httplib.responses[httplib.OK])
            else:
                body = self.fixtures.load('initiate_multipart.xml')
                return (httplib.OK, body, headers, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])
        else:
            headers = {'etag': '"0cc175b9c0f1b6a831c399e269772661"'}
            return (httplib.OK, '', headers, httplib.responses[httplib.OK])

    def _foo_bar_container_LIST_MULTIPART(self, method, url, body, headers):
        query_string = urlparse.urlsplit(url).query
        query = parse_qs(query_string)
        if 'key-marker' not in query:
            body = self.fixtures.load('list_multipart_1.xml')
        else:
            body = self.fixtures.load('list_multipart_2.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_my_divisor_LIST_MULTIPART(self, method, url, body, headers):
        body = ''
        return (httplib.NO_CONTENT, body, headers, httplib.responses[httplib.NO_CONTENT])

    def _foo_bar_container_my_movie_m2ts_LIST_MULTIPART(self, method, url, body, headers):
        body = ''
        return (httplib.NO_CONTENT, body, headers, httplib.responses[httplib.NO_CONTENT])

    def parse_body(self):
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        try:
            try:
                body = ET.XML(self.body)
            except ValueError:
                body = ET.XML(self.body.encode('utf-8'))
        except Exception:
            raise MalformedResponseError('Failed to parse XML', body=self.body, driver=self.connection.driver)
        return body

    def _foo_bar_container_foo_bar_object(self, method, url, body, headers):
        body = generate_random_data(1000)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_bar_object_range(self, method, url, body, headers):
        body = '0123456789123456789'
        self.assertTrue('Range' in headers)
        self.assertEqual(headers['Range'], 'bytes=5-6')
        start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['Range'], body)
        return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], headers, httplib.responses[httplib.PARTIAL_CONTENT])

    def _foo_bar_container_foo_bar_object_range_stream(self, method, url, body, headers):
        body = '0123456789123456789'
        self.assertTrue('Range' in headers)
        self.assertEqual(headers['Range'], 'bytes=4-6')
        start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['Range'], body)
        return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], headers, httplib.responses[httplib.PARTIAL_CONTENT])

    def _foo_bar_container_foo_bar_object_NO_BUFFER(self, method, url, body, headers):
        body = generate_random_data(1000)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_test_upload(self, method, url, body, headers):
        body = ''
        headers = {'etag': '"0cc175b9c0f1b6a831c399e269772661"'}
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _foo_bar_container_foo_bar_object_INVALID_SIZE(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])