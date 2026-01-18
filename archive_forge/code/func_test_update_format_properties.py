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
def test_update_format_properties(self):
    statuses_for_immutability = ['active', 'saving', 'killed']
    request = unit_test_utils.get_fake_request(roles=['admin'], is_admin=True)
    for status in statuses_for_immutability:
        image = {'id': str(uuid.uuid4()), 'status': status, 'disk_format': 'ari', 'container_format': 'ari'}
        self.db.image_create(None, image)
        changes = [{'op': 'replace', 'path': ['disk_format'], 'value': 'ami'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image['id'], changes)
        changes = [{'op': 'replace', 'path': ['container_format'], 'value': 'ami'}]
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image['id'], changes)
    self.db.image_update(None, image['id'], {'status': 'queued'})
    changes = [{'op': 'replace', 'path': ['disk_format'], 'value': 'raw'}, {'op': 'replace', 'path': ['container_format'], 'value': 'bare'}]
    resp = self.controller.update(request, image['id'], changes)
    self.assertEqual('raw', resp.disk_format)
    self.assertEqual('bare', resp.container_format)