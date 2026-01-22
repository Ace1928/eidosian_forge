import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
class TestImagesDeserializerWithAdditionalProperties(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagesDeserializerWithAdditionalProperties, self).setUp()
        self.config(allow_additional_image_properties=True)
        self.deserializer = glance.api.v2.images.RequestDeserializer()

    def test_create(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'foo': 'bar'})
        output = self.deserializer.create(request)
        expected = {'image': {}, 'extra_properties': {'foo': 'bar'}, 'tags': []}
        self.assertEqual(expected, output)

    def test_create_with_numeric_property(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'abc': 123})
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_update_with_numeric_property(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/openstack-images-v2.1-json-patch'
        doc = [{'op': 'add', 'path': '/foo', 'value': 123}]
        request.body = jsonutils.dump_as_bytes(doc)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_create_with_list_property(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'foo': ['bar']})
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_update_with_list_property(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/openstack-images-v2.1-json-patch'
        doc = [{'op': 'add', 'path': '/foo', 'value': ['bar', 'baz']}]
        request.body = jsonutils.dump_as_bytes(doc)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)

    def test_update(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/openstack-images-v2.1-json-patch'
        doc = [{'op': 'add', 'path': '/foo', 'value': 'bar'}]
        request.body = jsonutils.dump_as_bytes(doc)
        output = self.deserializer.update(request)
        change = {'json_schema_version': 10, 'op': 'add', 'path': ['foo'], 'value': 'bar'}
        self.assertEqual({'changes': [change]}, output)