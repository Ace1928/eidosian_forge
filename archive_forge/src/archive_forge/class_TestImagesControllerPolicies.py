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
class TestImagesControllerPolicies(base.IsolatedUnitTest):

    def setUp(self):
        super(TestImagesControllerPolicies, self).setUp()
        self.db = unit_test_utils.FakeDB()
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.controller = glance.api.v2.images.ImagesController(self.db, self.policy)
        store = unit_test_utils.FakeStoreAPI()
        self.store_utils = unit_test_utils.FakeStoreUtils(store)

    def test_index_unauthorized(self):
        rules = {'get_images': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.index, request)

    def test_show_unauthorized(self):
        rules = {'get_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.show, request, image_id=UUID2)

    def test_create_image_unauthorized(self):
        rules = {'add_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-1'}
        extra_properties = {}
        tags = []
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image, extra_properties, tags)

    def test_create_public_image_unauthorized(self):
        rules = {'publicize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-1', 'visibility': 'public'}
        extra_properties = {}
        tags = []
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image, extra_properties, tags)

    def test_create_community_image_unauthorized(self):
        rules = {'communitize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        image = {'name': 'image-c1', 'visibility': 'community'}
        extra_properties = {}
        tags = []
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image, extra_properties, tags)

    def test_update_unauthorized(self):
        rules = {'modify_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['name'], 'value': 'image-2'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_publicize_image_unauthorized(self):
        rules = {'publicize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['visibility'], 'value': 'public'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_communitize_image_unauthorized(self):
        rules = {'communitize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['visibility'], 'value': 'community'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_depublicize_image_unauthorized(self):
        rules = {'publicize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['visibility'], 'value': 'private'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual('private', output.visibility)

    def test_update_decommunitize_image_unauthorized(self):
        rules = {'communitize_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['visibility'], 'value': 'private'}]
        output = self.controller.update(request, UUID1, changes)
        self.assertEqual('private', output.visibility)

    def test_update_get_image_location_unauthorized(self):
        rules = {'get_image_location': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': []}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_set_image_location_unauthorized(self):

        def fake_delete_image_location_from_backend(self, *args, **kwargs):
            pass
        rules = {'set_image_location': False}
        self.policy.set_rules(rules)
        new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_update_delete_image_location_unauthorized(self):
        rules = {'delete_image_location': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        changes = [{'op': 'replace', 'path': ['locations'], 'value': []}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID1, changes)

    def test_delete_unauthorized(self):
        rules = {'delete_image': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID1)