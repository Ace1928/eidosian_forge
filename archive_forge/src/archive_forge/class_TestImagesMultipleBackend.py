import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
class TestImagesMultipleBackend(functional.MultipleBackendFunctionalTest):

    def setUp(self):
        super(TestImagesMultipleBackend, self).setUp()
        self.cleanup()
        self.include_scrubber = False
        self.api_server_multiple_backend.deployment_flavor = 'noauth'
        for i in range(3):
            ret = test_utils.start_http_server('foo_image_id%d' % i, 'foo_image%d' % i)
            setattr(self, 'http_server%d' % i, ret[1])
            setattr(self, 'http_port%d' % i, ret[2])

    def tearDown(self):
        for i in range(3):
            httpd = getattr(self, 'http_server%d' % i, None)
            if httpd:
                httpd.shutdown()
                httpd.server_close()
        super(TestImagesMultipleBackend, self).tearDown()

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_image_import_using_glance_direct(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('glance-direct', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        image_data = b'QQQQQ'
        path = self._url('/v2/images/%s/stage' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        func_utils.verify_image_hashes_and_status(self, image_id, size=len(image_data), status='uploading')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'method': {'name': 'glance-direct'}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=15, delay_sec=0.2)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(len(image_data), jsonutils.loads(response.text)['size'])
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_using_glance_direct_different_backend(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('glance-direct', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        image_data = b'GLANCE IS DEAD SEXY'
        path = self._url('/v2/images/%s/stage' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        func_utils.verify_image_hashes_and_status(self, image_id, size=len(image_data), status='uploading')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Image-Meta-Store': 'file2'})
        data = jsonutils.dumps({'method': {'name': 'glance-direct'}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=15, delay_sec=0.2)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(len(image_data), jsonutils.loads(response.text)['size'])
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_using_web_download(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('web-download', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=20, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_using_web_download_different_backend(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('web-download', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin', 'X-Image-Meta-Store': 'file2'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=20, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_multi_stores(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('web-download', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}, 'stores': ['file1', 'file2']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=40, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_copy_image_lifecycle(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('copy-image', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}, 'stores': ['file1']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        import_reqid = response.headers['X-Openstack-Request-Id']
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=40, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/tasks' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tasks = jsonutils.loads(response.text)['tasks']
        self.assertEqual(1, len(tasks))
        for task in tasks:
            self.assertEqual(image_id, task['image_id'])
            user_id = response.request.headers.get('X-User-Id')
            self.assertEqual(user_id, task['user_id'])
            self.assertEqual(import_reqid, task['request_id'])
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2', 'file3']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        copy_reqid = response.headers['X-Openstack-Request-Id']
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_copying(request_path=path, request_headers=self._headers(), stores=['file2', 'file3'], max_sec=40, delay_sec=0.2, start_delay_sec=1)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        self.assertIn('file3', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/tasks' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tasks = jsonutils.loads(response.text)['tasks']
        self.assertEqual(2, len(tasks))
        expected_reqids = [copy_reqid, import_reqid]
        for task in tasks:
            self.assertEqual(image_id, task['image_id'])
            user_id = response.request.headers.get('X-User-Id')
            self.assertEqual(user_id, task['user_id'])
            self.assertEqual(expected_reqids.pop(), task['request_id'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_copy_image_revert_lifecycle(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('copy-image', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}, 'stores': ['file1']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=40, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        os.rmdir(self.test_dir + '/images_3')
        data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2', 'file3']})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)

        def poll_callback(image):
            return not (image['os_glance_importing_to_stores'] == '' and image['os_glance_failed_import'] == 'file3' and (image['stores'] == 'file1'))
        func_utils.poll_entity(self._url('/v2/images/%s' % image_id), self._headers(), poll_callback)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        self.assertNotIn('file2', jsonutils.loads(response.text)['stores'])
        self.assertNotIn('file3', jsonutils.loads(response.text)['stores'])
        fail_key = 'os_glance_failed_import'
        pend_key = 'os_glance_importing_to_stores'
        self.assertEqual('file3', jsonutils.loads(response.text)[fail_key])
        self.assertEqual('', jsonutils.loads(response.text)[pend_key])
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2', 'file3'], 'all_stores_must_succeed': False})
        for i in range(0, 5):
            response = requests.post(path, headers=headers, data=data)
            if response.status_code != http.CONFLICT:
                break
            time.sleep(1)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_copying(request_path=path, request_headers=self._headers(), stores=['file2'], max_sec=10, delay_sec=0.2, start_delay_sec=1, failure_scenario=True)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        self.assertNotIn('file3', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_import_multi_stores_specifying_all_stores(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        path = self._url('/v2/info/import')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['import-methods']['value']
        self.assertIn('web-download', discovery_calls)
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        func_utils.verify_image_hashes_and_status(self, image_id, status='queued')
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
        thread, httpd, port = test_utils.start_standalone_http_server()
        image_data_uri = 'http://localhost:%s/' % port
        data = jsonutils.dumps({'method': {'name': 'web-download', 'uri': image_data_uri}, 'all_stores': True})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_status(request_path=path, request_headers=self._headers(), status='active', max_sec=40, delay_sec=0.2, start_delay_sec=1)
        with requests.get(image_data_uri) as r:
            expect_c = str(md5(r.content, usedforsecurity=False).hexdigest())
            expect_h = str(hashlib.sha512(r.content).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(r.content), status='active')
        httpd.shutdown()
        httpd.server_close()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file3', jsonutils.loads(response.text)['stores'])
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_lifecycle(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'abc': 'xyz', 'protected': True})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'foo', 'abc', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': True, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'foo': 'bar', 'abc': 'xyz', 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        image_data = b'OpenStack Rules, Other Clouds Drool'
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(expect_c, response.headers['Content-MD5'])
        self.assertEqual(image_data.decode('utf-8'), response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(len(image_data), jsonutils.loads(response.text)['size'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        doc = [{'op': 'replace', 'path': '/protected', 'value': False}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()

    def test_image_lifecycle_different_backend(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        available_stores = ['file1', 'file2', 'file3']
        path = self._url('/v2/info/stores')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        discovery_calls = jsonutils.loads(response.text)['stores']
        for stores in discovery_calls:
            self.assertIn('id', stores)
            self.assertIn(stores['id'], available_stores)
            self.assertFalse(stores['id'].startswith('os_glance_'))
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'abc': 'xyz', 'protected': True})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        self.assertIn('OpenStack-image-store-ids', response.headers)
        for store in available_stores:
            self.assertIn(store, response.headers['OpenStack-image-store-ids'])
        image = jsonutils.loads(response.text)
        image_id = image['id']
        checked_keys = set(['status', 'name', 'tags', 'created_at', 'updated_at', 'visibility', 'self', 'protected', 'id', 'file', 'min_disk', 'foo', 'abc', 'type', 'min_ram', 'schema', 'disk_format', 'container_format', 'owner', 'checksum', 'size', 'virtual_size', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
        self.assertEqual(checked_keys, set(image.keys()))
        expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': True, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'foo': 'bar', 'abc': 'xyz', 'type': 'kernel', 'min_ram': 0, 'schema': '/v2/schemas/image'}
        for key, value in expected_image.items():
            self.assertEqual(value, image[key], key)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(1, len(images))
        self.assertEqual(image_id, images[0]['id'])
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        image_data = b'just a passing glance'
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers({'Content-Type': 'application/octet-stream', 'X-Image-Meta-Store': 'file2'})
        response = requests.put(path, headers=headers, data=image_data)
        self.assertEqual(http.NO_CONTENT, response.status_code)
        expect_c = str(md5(image_data, usedforsecurity=False).hexdigest())
        expect_h = str(hashlib.sha512(image_data).hexdigest())
        func_utils.verify_image_hashes_and_status(self, image_id, checksum=expect_c, os_hash_value=expect_h, size=len(image_data), status='active')
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/file' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(expect_c, response.headers['Content-MD5'])
        self.assertEqual(image_data.decode('utf-8'), response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(len(image_data), jsonutils.loads(response.text)['size'])
        path = self._url('/v2/images/%s' % image_id)
        media_type = 'application/openstack-images-v2.1-json-patch'
        headers = self._headers({'content-type': media_type})
        doc = [{'op': 'replace', 'path': '/protected', 'value': False}]
        data = jsonutils.dumps(doc)
        response = requests.patch(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images/%s/file' % image_id)
        headers = self._headers()
        response = requests.get(path, headers=headers)
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/images')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))
        self.stop_servers()