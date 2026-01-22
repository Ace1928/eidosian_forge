import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
class TestCreatingLocations(base.MultiStoreBaseTest):
    _CONF = cfg.ConfigOpts()

    def setUp(self):
        super(TestCreatingLocations, self).setUp()
        enabled_backends = {'swift1': 'swift', 'swift2': 'swift'}
        self.conf = self._CONF
        self.conf(args=[])
        self.conf.register_opt(cfg.DictOpt('enabled_backends'))
        self.config(enabled_backends=enabled_backends)
        store.register_store_opts(self.conf)
        self.config(default_backend='swift1', group='glance_store')
        location.SCHEME_TO_CLS_BACKEND_MAP = {}
        store.create_multi_stores(self.conf)
        self.addCleanup(setattr, location, 'SCHEME_TO_CLS_BACKEND_MAP', dict())
        self.test_dir = self.useFixture(fixtures.TempDir()).path
        config = copy.deepcopy(SWIFT_CONF)
        self.store = Store(self.conf, backend='swift1')
        self.config(group='swift1', **config)
        self.store.configure()
        self.register_store_backend_schemes(self.store, 'swift', 'swift1')
        importlib.reload(swift)
        self.addCleanup(self.conf.reset)
        service_catalog = [{'endpoint_links': [], 'endpoints': [{'adminURL': 'https://some_admin_endpoint', 'region': 'RegionOne', 'internalURL': 'https://some_internal_endpoint', 'publicURL': 'https://some_endpoint'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        self.ctxt = mock.MagicMock(user='user', tenant='tenant', auth_token='123', service_catalog=service_catalog)

    def test_single_tenant_location(self):
        conf = copy.deepcopy(SWIFT_CONF)
        conf['swift_store_container'] = 'container'
        conf_file = 'glance-swift.conf'
        self.swift_config_file = self.copy_data_file(conf_file, self.test_dir)
        conf.update({'swift_store_config_file': self.swift_config_file})
        conf['default_swift_reference'] = 'ref1'
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        store = swift.SingleTenantStore(self.conf, backend='swift1')
        store.configure()
        location = store.create_location('image-id')
        self.assertEqual('swift+https', location.scheme)
        self.assertEqual('https://example.com', location.swift_url)
        self.assertEqual('container', location.container)
        self.assertEqual('image-id', location.obj)
        self.assertEqual('tenant:user1', location.user)
        self.assertEqual('key1', location.key)

    def test_single_tenant_location_http(self):
        conf_file = 'glance-swift.conf'
        test_dir = self.useFixture(fixtures.TempDir()).path
        self.swift_config_file = self.copy_data_file(conf_file, test_dir)
        self.config(group='swift1', swift_store_container='container', default_swift_reference='ref2', swift_store_config_file=self.swift_config_file)
        store = swift.SingleTenantStore(self.conf, backend='swift1')
        store.configure()
        location = store.create_location('image-id')
        self.assertEqual('swift+http', location.scheme)
        self.assertEqual('http://example.com', location.swift_url)

    def test_multi_tenant_location(self):
        self.config(group='swift1', swift_store_container='container')
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        location = store.create_location('image-id', context=self.ctxt)
        self.assertEqual('swift+https', location.scheme)
        self.assertEqual('https://some_endpoint', location.swift_url)
        self.assertEqual('container_image-id', location.container)
        self.assertEqual('image-id', location.obj)
        self.assertIsNone(location.user)
        self.assertIsNone(location.key)

    def test_multi_tenant_location_http(self):
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        self.ctxt.service_catalog[0]['endpoints'][0]['publicURL'] = 'http://some_endpoint'
        location = store.create_location('image-id', context=self.ctxt)
        self.assertEqual('swift+http', location.scheme)
        self.assertEqual('http://some_endpoint', location.swift_url)

    def test_multi_tenant_location_with_region(self):
        self.config(group='swift1', swift_store_region='WestCarolina')
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        self.ctxt.service_catalog[0]['endpoints'][0]['region'] = 'WestCarolina'
        self.assertEqual('https://some_endpoint', store._get_endpoint(self.ctxt))

    def test_multi_tenant_location_custom_service_type(self):
        self.config(group='swift1', swift_store_service_type='toy-store')
        self.ctxt.service_catalog[0]['type'] = 'toy-store'
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        store._get_endpoint(self.ctxt)
        self.assertEqual('https://some_endpoint', store._get_endpoint(self.ctxt))

    def test_multi_tenant_location_custom_endpoint_type(self):
        self.config(group='swift1', swift_store_endpoint_type='internalURL')
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        self.assertEqual('https://some_internal_endpoint', store._get_endpoint(self.ctxt))