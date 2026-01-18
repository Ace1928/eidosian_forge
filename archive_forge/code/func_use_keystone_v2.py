import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def use_keystone_v2(self):
    self.adapter = self.useFixture(rm_fixture.Fixture())
    self.calls = []
    self._uri_registry.clear()
    self.__do_register_uris([self.get_keystone_discovery(), dict(method='POST', uri='https://identity.example.com/v2.0/tokens', json=self.os_fixture.v2_token)])
    self._make_test_cloud(cloud_name='_test_cloud_v2_', identity_api_version='2.0')