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
def test_set_container_temp_url_key_second(self):
    key = 'super-secure-key'
    self.register_uris([dict(method='POST', uri=self.container_endpoint, status_code=204, validate=dict(headers={'x-container-meta-temp-url-key-2': key})), dict(method='HEAD', uri=self.container_endpoint, headers={'x-container-meta-temp-url-key-2': key})])
    self.proxy.set_container_temp_url_key(self.container, key, secondary=True)
    self.assert_calls()