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
def test_property_protections_with_policies(self):
    rules = {'glance_creator': 'role:admin or role:spl_role'}
    self.set_policy_rules(rules)
    self.api_server.property_protection_file = self.property_file_policies
    self.api_server.property_protection_rule_format = 'policies'
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    data = jsonutils.dumps({'name': 'image-1', 'foo': 'bar', 'disk_format': 'aki', 'container_format': 'aki', 'x_owner_foo': 'o_s_bar'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    expected_image = {'status': 'queued', 'name': 'image-1', 'tags': [], 'visibility': 'shared', 'self': '/v2/images/%s' % image_id, 'protected': False, 'file': '/v2/images/%s/file' % image_id, 'min_disk': 0, 'min_ram': 0, 'schema': '/v2/schemas/image'}
    for key, value in expected_image.items():
        self.assertEqual(value, image[key], key)
    path = self._url('/v2/images')
    headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member,spl_role, admin'})
    data = jsonutils.dumps({'name': 'image-1', 'disk_format': 'aki', 'container_format': 'aki', 'spl_creator_policy': 'creator_bar', 'spl_default_policy': 'default_bar'})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image = jsonutils.loads(response.text)
    image_id = image['id']
    self.assertEqual('creator_bar', image['spl_creator_policy'])
    self.assertEqual('default_bar', image['spl_default_policy'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': ''}, {'op': 'replace', 'path': '/spl_creator_policy', 'value': 'r'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertEqual('r', image['spl_creator_policy'])
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,member,spl_role'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': 'z'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.FORBIDDEN, response.status_code, response.text)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,random_role'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertEqual(image['spl_default_policy'], 'default_bar')
    self.assertNotIn('spl_creator_policy', image)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type, 'X-Roles': 'admin'})
    data = jsonutils.dumps([{'op': 'replace', 'path': '/spl_creator_policy', 'value': ''}, {'op': 'remove', 'path': '/spl_creator_policy'}])
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    image = jsonutils.loads(response.text)
    self.assertNotIn('spl_creator_policy', image)
    path = self._url('/v2/images/%s' % image_id)
    headers = self._headers({'content-type': media_type, 'X-Roles': 'reader,random_role'})
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    image = jsonutils.loads(response.text)
    self.assertEqual(image['spl_default_policy'], 'default_bar')
    path = self._url('/v2/images/%s' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    self.stop_servers()