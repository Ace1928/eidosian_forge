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
def _test_update_locations_status(self, image_status, update):
    self.config(show_multiple_locations=True)
    self.images = [_db_fixture('1', owner=TENANT1, checksum=CHKSUM, name='1', disk_format='raw', container_format='bare', status=image_status)]
    request = unit_test_utils.get_fake_request()
    if image_status == 'deactivated':
        self.db.image_create(request.context, self.images[0])
    else:
        self.db.image_create(None, self.images[0])
    new_location = {'url': '%s/fake_location' % BASE_URI, 'metadata': {}}
    changes = [{'op': update, 'path': ['locations', '-'], 'value': new_location}]
    self.assertRaises(webob.exc.HTTPConflict, self.controller.update, request, '1', changes)