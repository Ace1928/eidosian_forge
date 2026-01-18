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
def test_update_no_changes(self):
    request = unit_test_utils.get_fake_request()
    output = self.controller.update(request, UUID1, changes=[])
    self.assertEqual(UUID1, output.image_id)
    self.assertEqual(output.created_at, output.updated_at)
    self.assertEqual(2, len(output.tags))
    self.assertIn('ping', output.tags)
    self.assertIn('pong', output.tags)
    output_logs = self.notifier.get_logs()
    self.assertEqual(0, len(output_logs))