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
def test_property_protections_special_chars_roles(self):
    self.api_server.property_protection_file = self.property_file_roles
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_admin': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_admin': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_joe_soap': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'x_all_permitted_joe_soap': '1', 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertEqual('1', image['x_all_permitted_joe_soap'])
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertEqual('1', image['x_all_permitted_joe_soap'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '2'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertEqual('2', image['x_all_permitted_joe_soap'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/x_all_permitted_joe_soap', 'value': '3'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertEqual('3', image['x_all_permitted_joe_soap'])
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_all_permitted_a': '1', 'x_all_permitted_b': '2'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_a'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertNotIn('x_all_permitted_a', image.keys())
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps([{'op': 'remove', 'path': '/x_all_permitted_b'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertNotIn('x_all_permitted_b', image.keys())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_admin': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_permitted_joe_soap': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_read': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    self.assertNotIn('x_none_read', image.keys())
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertNotIn('x_none_read', image.keys())
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,joe_soap'})
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertNotIn('x_none_read', image.keys())
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_update': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    self.assertEqual('1', image['x_none_update'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '2'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/x_none_update', 'value': '3'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'x_none_delete': '1'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,joe_soap'})
    data = jsonutils.dumps([{'op': 'remove', 'path': '/x_none_delete'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    self.stop_servers()