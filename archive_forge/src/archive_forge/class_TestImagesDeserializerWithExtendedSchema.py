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
class TestImagesDeserializerWithExtendedSchema(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagesDeserializerWithExtendedSchema, self).setUp()
        self.config(allow_additional_image_properties=False)
        custom_image_properties = {'pants': {'type': 'string', 'enum': ['on', 'off']}}
        schema = glance.api.v2.images.get_schema(custom_image_properties)
        self.deserializer = glance.api.v2.images.RequestDeserializer(schema)

    def test_create(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'name': 'image-1', 'pants': 'on'})
        output = self.deserializer.create(request)
        expected = {'image': {'name': 'image-1'}, 'extra_properties': {'pants': 'on'}, 'tags': []}
        self.assertEqual(expected, output)

    def test_create_bad_data(self):
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes({'name': 'image-1', 'pants': 'borked'})
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_update(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/openstack-images-v2.1-json-patch'
        doc = [{'op': 'add', 'path': '/pants', 'value': 'off'}]
        request.body = jsonutils.dump_as_bytes(doc)
        output = self.deserializer.update(request)
        expected = {'changes': [{'json_schema_version': 10, 'op': 'add', 'path': ['pants'], 'value': 'off'}]}
        self.assertEqual(expected, output)

    def test_update_bad_data(self):
        request = unit_test_utils.get_fake_request()
        request.content_type = 'application/openstack-images-v2.1-json-patch'
        doc = [{'op': 'add', 'path': '/pants', 'value': 'cutoffs'}]
        request.body = jsonutils.dump_as_bytes(doc)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)