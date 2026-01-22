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
class TestCopyImagePermissions(functional.MultipleBackendFunctionalTest):

    def setUp(self):
        super(TestCopyImagePermissions, self).setUp()
        self.cleanup()
        self.include_scrubber = False
        self.api_server_multiple_backend.deployment_flavor = 'noauth'

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def _create_and_import_image_data(self):
        path = self._url('/v2/images')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'name': 'image-1', 'type': 'kernel', 'visibility': 'public', 'disk_format': 'aki', 'container_format': 'aki'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        image = jsonutils.loads(response.text)
        image_id = image['id']
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
        return image_id

    def _test_copy_public_image_as_non_admin(self):
        self.start_servers(**self.__dict__.copy())
        image_id = self._create_and_import_image_data()
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual('file1', jsonutils.loads(response.text)['stores'])
        path = self._url('/v2/images/%s/import' % image_id)
        headers = self._headers({'content-type': 'application/json'})
        headers = get_auth_header(TENANT2, TENANT2, role='reader,member', headers=headers)
        data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2']})
        response = requests.post(path, headers=headers, data=data)
        return (image_id, response)

    def test_copy_public_image_as_non_admin(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': '', 'download_image': '', 'add_member': '', 'publicize_image': '', 'copy_image': 'role:admin'}
        self.set_policy_rules(rules)
        image_id, response = self._test_copy_public_image_as_non_admin()
        self.assertEqual(http.FORBIDDEN, response.status_code)

    def test_copy_public_image_as_non_admin_permitted(self):
        rules = {'context_is_admin': 'role:admin', 'default': '', 'add_image': '', 'get_image': '', 'modify_image': '', 'upload_image': '', 'get_image_location': '', 'delete_image': '', 'restricted': '', 'download_image': '', 'add_member': '', 'publicize_image': '', 'copy_image': "'public':%(visibility)s"}
        self.set_policy_rules(rules)
        image_id, response = self._test_copy_public_image_as_non_admin()
        self.assertEqual(http.ACCEPTED, response.status_code)
        path = self._url('/v2/images/%s' % image_id)
        func_utils.wait_for_copying(request_path=path, request_headers=self._headers(), stores=['file2'], max_sec=40, delay_sec=0.2, start_delay_sec=1)
        path = self._url('/v2/images/%s' % image_id)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertIn('file2', jsonutils.loads(response.text)['stores'])