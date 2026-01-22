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
class IronicTestCase(TestCase):

    def setUp(self):
        super(IronicTestCase, self).setUp()
        self.use_ironic()
        self.uuid = str(uuid.uuid4())
        self.name = self.getUniqueString('name')

    def get_mock_url(self, **kwargs):
        kwargs.setdefault('service_type', 'baremetal')
        kwargs.setdefault('interface', 'public')
        kwargs.setdefault('base_url_append', 'v1')
        return super(IronicTestCase, self).get_mock_url(**kwargs)