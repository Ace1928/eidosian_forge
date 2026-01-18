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
def test_update_v2_0_compatibility(self):
    request = self._get_fake_patch_request(content_type_minor_version=0)
    body = [{'replace': '/name', 'value': 'fedora'}, {'replace': '/tags', 'value': ['king', 'kong']}, {'replace': '/foo', 'value': 'bar'}, {'add': '/bebim', 'value': 'bap'}, {'remove': '/sparks'}, {'add': '/locations/-', 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'add': '/locations/10', 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'remove': '/locations/2'}, {'replace': '/locations', 'value': []}, {'replace': '/locations', 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]
    request.body = jsonutils.dump_as_bytes(body)
    output = self.deserializer.update(request)
    expected = {'changes': [{'json_schema_version': 4, 'op': 'replace', 'path': ['name'], 'value': 'fedora'}, {'json_schema_version': 4, 'op': 'replace', 'path': ['tags'], 'value': ['king', 'kong']}, {'json_schema_version': 4, 'op': 'replace', 'path': ['foo'], 'value': 'bar'}, {'json_schema_version': 4, 'op': 'add', 'path': ['bebim'], 'value': 'bap'}, {'json_schema_version': 4, 'op': 'remove', 'path': ['sparks']}, {'json_schema_version': 4, 'op': 'add', 'path': ['locations', '-'], 'value': {'url': 'scheme3://path3', 'metadata': {}}}, {'json_schema_version': 4, 'op': 'add', 'path': ['locations', '10'], 'value': {'url': 'scheme4://path4', 'metadata': {}}}, {'json_schema_version': 4, 'op': 'remove', 'path': ['locations', '2']}, {'json_schema_version': 4, 'op': 'replace', 'path': ['locations'], 'value': []}, {'json_schema_version': 4, 'op': 'replace', 'path': ['locations'], 'value': [{'url': 'scheme5://path5', 'metadata': {}}, {'url': 'scheme6://path6', 'metadata': {}}]}]}
    self.assertEqual(expected, output)