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
def test_tag_lifecycle(self):
    self.start_servers(**self.__dict__.copy())
    path = self._url('/v2/images')
    headers = self._headers({'Content-Type': 'application/json'})
    data = jsonutils.dumps({'name': 'image-1', 'tags': ['sniff', 'sniff']})
    response = requests.post(path, headers=headers, data=data)
    self.assertEqual(http.CREATED, response.status_code)
    image_id = jsonutils.loads(response.text)['id']
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['sniff'], tags)
    for tag in tags:
        path = self._url('/v2/images/%s/tags/%s' % (image_id, tag))
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
    for i in range(10):
        path = self._url('/v2/images/%s/tags/foo%i' % (image_id, i))
        response = requests.put(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s/tags/fail_me' % image_id)
    response = requests.put(path, headers=self._headers())
    self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(10, len(tags))
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    doc = [{'op': 'replace', 'path': '/tags', 'value': ['foo']}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    tags = ['foo%d' % i for i in range(11)]
    doc = [{'op': 'replace', 'path': '/tags', 'value': tags}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.REQUEST_ENTITY_TOO_LARGE, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['foo'], tags)
    path = self._url('/v2/images/%s' % image_id)
    media_type = 'application/openstack-images-v2.1-json-patch'
    headers = self._headers({'content-type': media_type})
    doc = [{'op': 'replace', 'path': '/tags', 'value': ['sniff', 'snozz', 'snozz']}]
    data = jsonutils.dumps(doc)
    response = requests.patch(path, headers=headers, data=data)
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['sniff', 'snozz'], sorted(tags))
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['sniff', 'snozz'], sorted(tags))
    path = self._url('/v2/images/%s/tags/snozz' % image_id)
    response = requests.put(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
    response = requests.put(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['gabe@example.com', 'sniff', 'snozz'], sorted(tags))
    path = self._url('/v2/images?tag=sniff')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual('image-1', images[0]['name'])
    path = self._url('/v2/images?tag=sniff&tag=snozz')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual('image-1', images[0]['name'])
    path = self._url('/v2/images?tag=sniff&status=queued')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(1, len(images))
    self.assertEqual('image-1', images[0]['name'])
    path = self._url('/v2/images?tag=sniff&tag=fake')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NO_CONTENT, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['sniff', 'snozz'], sorted(tags))
    path = self._url('/v2/images/%s/tags/gabe%%40example.com' % image_id)
    response = requests.delete(path, headers=self._headers())
    self.assertEqual(http.NOT_FOUND, response.status_code)
    path = self._url('/v2/images?tag=gabe%%40example.com')
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
    big_tag = 'a' * 300
    path = self._url('/v2/images/%s/tags/%s' % (image_id, big_tag))
    response = requests.put(path, headers=self._headers())
    self.assertEqual(http.BAD_REQUEST, response.status_code)
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    tags = jsonutils.loads(response.text)['tags']
    self.assertEqual(['sniff', 'snozz'], sorted(tags))
    self.stop_servers()