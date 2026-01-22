import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
class TestOsloConfig(_TestConnectionBase):

    def test_from_conf(self):
        c1 = connection.Connection(cloud='sample-cloud')
        conn = connection.Connection(session=c1.session, oslo_conf=self._load_ks_cfg_opts())
        self.assertIsInstance(conn.identity, service_description._ServiceDisabledProxyShim)
        self.assertEqual('openstack.compute.v2._proxy', conn.compute.__class__.__module__)

    def test_from_conf_filter_service_types(self):
        c1 = connection.Connection(cloud='sample-cloud')
        conn = connection.Connection(session=c1.session, oslo_conf=self._load_ks_cfg_opts(), service_types={'orchestration', 'i-am-ignored'})
        self.assertIsInstance(conn.identity, service_description._ServiceDisabledProxyShim)
        self.assertIsInstance(conn.compute, service_description._ServiceDisabledProxyShim)