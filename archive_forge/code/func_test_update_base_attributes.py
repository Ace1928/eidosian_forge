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
def test_update_base_attributes(self):
    request = self._get_fake_patch_request()
    body = [{'op': 'replace', 'path': '/name', 'value': 'fedora'}, {'op': 'replace', 'path': '/visibility', 'value': 'public'}, {'op': 'replace', 'path': '/tags', 'value': ['king', 'kong']}, {'op': 'replace', 'path': '/protected', 'value': True}, {'op': 'replace', 'path': '/container_format', 'value': 'bare'}, {'op': 'replace', 'path': '/disk_format', 'value': 'raw'}, {'op': 'replace', 'path': '/min_ram', 'value': 128}, {'op': 'replace', 'path': '/min_disk', 'value': 10}, {'op': 'replace', 'path': '/locations', 'value': []}, {'op': 'replace', 'path': '/locations', 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]
    request.body = jsonutils.dump_as_bytes(body)
    output = self.deserializer.update(request)
    expected = {'changes': [{'json_schema_version': 10, 'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['visibility'], 'value': 'public'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}, {'json_schema_version': 10, 'op': 'replace', 'path': ['protected'], 'value': True}, {'json_schema_version': 10, 'op': 'replace', 'path': ['container_format'], 'value': 'bare'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['disk_format'], 'value': 'raw'}, {'json_schema_version': 10, 'op': 'replace', 'path': ['min_ram'], 'value': 128}, {'json_schema_version': 10, 'op': 'replace', 'path': ['min_disk'], 'value': 10}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': []}, {'json_schema_version': 10, 'op': 'replace', 'path': ['locations'], 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]}
    self.assertEqual(expected, output)