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
def test_prop_protection_with_create_and_unpermitted_role(self):
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
    self.set_property_protections()
    request = unit_test_utils.get_fake_request(roles=['admin'])
    image = {'name': 'image-1'}
    created_image = self.controller.create(request, image=image, extra_properties={}, tags=[])
    roles = ['fake_member']
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:fake_member'})
    self.controller.policy = enforcer
    another_request = unit_test_utils.get_fake_request(roles=roles)
    changes = [{'op': 'add', 'path': ['x_owner_foo'], 'value': 'bar'}]
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)