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
def test_object_create_attrs(self):
    kwargs = {'name': 'test', 'data': 'data', 'container': 'name', 'metadata': {}}
    self._verify('openstack.proxy.Proxy._create', self.proxy.upload_object, method_kwargs=kwargs, expected_args=[obj.Object], expected_kwargs=kwargs)