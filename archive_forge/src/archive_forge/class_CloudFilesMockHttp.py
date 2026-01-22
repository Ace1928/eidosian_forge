import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
class CloudFilesMockHttp(BaseRangeDownloadMockHttp, unittest.TestCase):
    fixtures = StorageFileFixtures('cloudfiles')
    base_headers = {'content-type': 'application/json; charset=UTF-8'}

    def _v2_0_tokens(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        body = self.fixtures.load('_v2_0__auth.json')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_MALFORMED_JSON(self, method, url, body, headers):
        body = 'broken: json /*"'
        return (httplib.NO_CONTENT, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_EMPTY(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        if method == 'GET':
            body = self.fixtures.load('list_containers.json')
            status_code = httplib.OK
        elif method == 'HEAD':
            body = self.fixtures.load('meta_data.json')
            status_code = httplib.NO_CONTENT
            headers.update({'x-account-container-count': '10', 'x-account-object-count': '400', 'x-account-bytes-used': '1234567'})
        elif method == 'POST':
            body = ''
            status_code = httplib.NO_CONTENT
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_not_found(self, method, url, body, headers):
        if method == 'HEAD':
            body = ''
        else:
            raise ValueError('Invalid method')
        return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects_empty.json')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_20container_201_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_container_objects_empty.json')
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        if method == 'GET':
            if url.find('marker') == -1:
                body = self.fixtures.load('list_container_objects.json')
                status_code = httplib.OK
            else:
                body = ''
                status_code = httplib.NO_CONTENT
        elif method == 'HEAD':
            body = self.fixtures.load('list_container_objects_empty.json')
            status_code = httplib.NO_CONTENT
            headers.update({'x-container-object-count': '800', 'x-container-bytes-used': '1234568'})
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container_ITERATOR(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        if url.find('foo-test-3') != -1:
            body = self.fixtures.load('list_container_objects_not_exhausted2.json')
            status_code = httplib.OK
        elif url.find('foo-test-5') != -1:
            body = ''
            status_code = httplib.NO_CONTENT
        else:
            body = self.fixtures.load('list_container_objects_not_exhausted1.json')
            status_code = httplib.OK
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container_not_found(self, method, url, body, headers):
        if method == 'HEAD':
            body = ''
        else:
            raise ValueError('Invalid method')
        return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container_test_object(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        if method == 'HEAD':
            body = self.fixtures.load('list_container_objects_empty.json')
            status_code = httplib.NO_CONTENT
            headers.update({'content-length': '555', 'last-modified': 'Tue, 25 Jan 2011 22:01:49 GMT', 'etag': '6b21c4a111ac178feacf9ec9d0c71f17', 'x-object-meta-foo-bar': 'test 1', 'x-object-meta-bar-foo': 'test 2', 'content-type': 'application/zip'})
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_container__7E_test_object(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        if method == 'HEAD':
            body = self.fixtures.load('list_container_objects_empty.json')
            status_code = httplib.NO_CONTENT
            headers.update({'content-length': '555', 'last-modified': 'Tue, 25 Jan 2011 22:01:49 GMT', 'etag': '6b21c4a111ac178feacf9ec9d0c71f17', 'x-object-meta-foo-bar': 'test 1', 'x-object-meta-bar-foo': 'test 2', 'content-type': 'application/zip'})
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_create_container(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        body = self.fixtures.load('list_container_objects_empty.json')
        headers = copy.deepcopy(self.base_headers)
        headers.update({'content-length': '18', 'date': 'Mon, 28 Feb 2011 07:52:57 GMT'})
        status_code = httplib.CREATED
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_speci_40l_name(self, method, url, body, headers):
        container_name = 'speci@l_name'
        encoded_container_name = urlquote(container_name)
        self.assertTrue(encoded_container_name in url)
        headers = copy.deepcopy(self.base_headers)
        body = self.fixtures.load('list_container_objects_empty.json')
        headers = copy.deepcopy(self.base_headers)
        headers.update({'content-length': '18', 'date': 'Mon, 28 Feb 2011 07:52:57 GMT'})
        status_code = httplib.CREATED
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_test_create_container_ALREADY_EXISTS(self, method, url, body, headers):
        headers = copy.deepcopy(self.base_headers)
        body = self.fixtures.load('list_container_objects_empty.json')
        headers.update({'content-type': 'text/plain'})
        status_code = httplib.ACCEPTED
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('list_container_objects_empty.json')
            headers = self.base_headers
            status_code = httplib.NO_CONTENT
        elif method == 'POST':
            body = ''
            headers = self.base_headers
            status_code = httplib.ACCEPTED
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_object_PURGE_SUCCESS(self, method, url, body, headers):
        if method == 'DELETE':
            headers = self.base_headers
            status_code = httplib.NO_CONTENT
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_object_PURGE_SUCCESS_EMAIL(self, method, url, body, headers):
        if method == 'DELETE':
            self.assertEqual(headers['X-Purge-Email'], 'test@test.com')
            headers = self.base_headers
            status_code = httplib.NO_CONTENT
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_NOT_FOUND(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('list_container_objects_empty.json')
            headers = self.base_headers
            status_code = httplib.NOT_FOUND
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_NOT_EMPTY(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('list_container_objects_empty.json')
            headers = self.base_headers
            status_code = httplib.CONFLICT
        return (status_code, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_bar_object(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('list_container_objects_empty.json')
            headers = self.base_headers
            status_code = httplib.NO_CONTENT
            return (status_code, body, headers, httplib.responses[httplib.OK])
        elif method == 'GET':
            body = generate_random_data(1000)
            return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_bar_object_range(self, method, url, body, headers):
        if method == 'GET':
            body = '0123456789123456789'
            self.assertTrue('Range' in headers)
            self.assertEqual(headers['Range'], 'bytes=5-6')
            start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['Range'], body)
            return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], self.base_headers, httplib.responses[httplib.PARTIAL_CONTENT])

    def _v1_MossoCloudFS_py3_img_or_vid(self, method, url, body, headers):
        headers = {'etag': 'e2378cace8712661ce7beec3d9362ef6'}
        headers.update(self.base_headers)
        return (httplib.CREATED, '', headers, httplib.responses[httplib.CREATED])

    def _v1_MossoCloudFS_foo_bar_container_foo_test_upload(self, method, url, body, headers):
        body = ''
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = 'hash343hhash89h932439jsaa89'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_speci_40l_name_m_40obj_E2_82_ACct(self, method, url, body, headers):
        object_name = 'm@obj€ct'
        urlquote(object_name)
        headers = copy.deepcopy(self.base_headers)
        body = ''
        headers['etag'] = 'hash343hhash89h932439jsaa89'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_empty(self, method, url, body, headers):
        body = ''
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = 'hash343hhash89h932439jsaa89'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_test_upload_INVALID_HASH(self, method, url, body, headers):
        body = ''
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = 'foobar'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_bar_object_INVALID_SIZE(self, method, url, body, headers):
        body = generate_random_data(100)
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_bar_object_NOT_FOUND(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_test_stream_data(self, method, url, body, headers):
        hasher = hashlib.md5()
        hasher.update(b'235')
        hash_value = hasher.hexdigest()
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = hash_value
        body = 'test'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_test_stream_data_seek(self, method, url, body, headers):
        hasher = hashlib.md5()
        hasher.update(b'123456789')
        hash_value = hasher.hexdigest()
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = hash_value
        body = 'test'
        return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])

    def _v1_MossoCloudFS_foo_bar_container_foo_bar_object_NO_BUFFER(self, method, url, body, headers):
        headers = {}
        headers.update(self.base_headers)
        headers['etag'] = '577ef1154f3240ad5b9b413aa7346a1e'
        body = generate_random_data(1000)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])