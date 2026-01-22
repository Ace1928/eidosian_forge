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
def _domain_fixture(id, **kwargs):
    properties = {'image_id': id, 'name': None, 'visibility': 'private', 'checksum': None, 'os_hash_algo': None, 'os_hash_value': None, 'owner': None, 'status': 'queued', 'size': None, 'virtual_size': None, 'locations': [], 'protected': False, 'disk_format': None, 'container_format': None, 'min_ram': None, 'min_disk': None, 'tags': []}
    properties.update(kwargs)
    return glance.domain.Image(**properties)