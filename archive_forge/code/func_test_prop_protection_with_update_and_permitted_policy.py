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
def test_prop_protection_with_update_and_permitted_policy(self):
    self.set_property_protections(use_policies=True)
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    self.controller = glance.api.v2.images.ImagesController(self.db, enforcer, self.notifier, self.store)
    request = unit_test_utils.get_fake_request(roles=['spl_role', 'admin'])
    image = {'name': 'image-1'}
    extra_props = {'spl_creator_policy': 'bar'}
    created_image = self.controller.create(request, image=image, extra_properties=extra_props, tags=[])
    self.assertEqual('bar', created_image.extra_properties['spl_creator_policy'])
    another_request = unit_test_utils.get_fake_request(roles=['spl_role'])
    changes = [{'op': 'replace', 'path': ['spl_creator_policy'], 'value': 'par'}]
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:spl_role'})
    self.controller.policy = enforcer
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, another_request, created_image.image_id, changes)
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': 'role:admin'})
    self.controller.policy = enforcer
    another_request = unit_test_utils.get_fake_request(roles=['admin'])
    output = self.controller.update(another_request, created_image.image_id, changes)
    self.assertEqual('par', output.extra_properties['spl_creator_policy'])