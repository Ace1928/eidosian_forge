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
def test_permissions(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'Content-Type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'raw', 'container_format': 'bare'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image_id = jsonutils.loads(response.text)['id']
    path = self._url('/v2/images/%s/file' % image_id)
    headers = self._headers({'Content-Type': 'application/octet-stream'})
    response = requests.put(path, headers=headers, data='ZZZZZ')
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'X-Tenant-Id': TENANT2})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'X-Tenant-Id': TENANT2})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT2})
    doc = [{'op': 'replace', 'path': '/name', 'value': 'image-2'}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'X-Tenant-Id': TENANT2})
    response = requests.delete(path, headers=headers)
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Roles': 'admin'})
    doc = [{'op': 'replace', 'path': '/visibility', 'value': 'public'}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'X-Tenant-Id': TENANT3})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(image_id, images[0]['id'])
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'X-Tenant-Id': TENANT3})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'Content-Type': 'application/openstack-images-v2.1-json-patch', 'X-Tenant-Id': TENANT3})
    doc = [{'op': 'replace', 'path': '/name', 'value': 'image-2'}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'X-Tenant-Id': TENANT3})
    response = requests.delete(path, headers=headers)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images/%s/file' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertEqual(response.text, 'ZZZZZ')
    self.stop_servers()