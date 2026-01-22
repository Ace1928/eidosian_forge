from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
class TestExtractName(TestObjectStoreProxy):
    scenarios = [('discovery', dict(url='/', parts=['account'])), ('endpoints', dict(url='/endpoints', parts=['endpoints'])), ('container', dict(url='/AUTH_123/container_name', parts=['container'])), ('object', dict(url='/container_name/object_name', parts=['object'])), ('object_long', dict(url='/v1/AUTH_123/cnt/path/deep/object_name', parts=['object']))]

    def test_extract_name(self):
        results = self.proxy._extract_name(self.url, project_id='123')
        self.assertEqual(self.parts, results)