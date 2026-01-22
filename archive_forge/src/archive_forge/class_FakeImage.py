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
class FakeImage(object):

    def __init__(self, id=None, status='active', container_format='ami', disk_format='ami', locations=None):
        self.id = id or UUID4
        self.status = status
        self.container_format = container_format
        self.disk_format = disk_format
        self.locations = locations
        self.owner = unit_test_utils.TENANT1
        self.created_at = ''
        self.updated_at = ''
        self.min_disk = ''
        self.min_ram = ''
        self.protected = False
        self.checksum = ''
        self.os_hash_algo = ''
        self.os_hash_value = ''
        self.size = 0
        self.virtual_size = 0
        self.visibility = 'public'
        self.os_hidden = False
        self.name = 'foo'
        self.tags = []
        self.extra_properties = {}
        self.member = self.owner
        self.image_id = self.id