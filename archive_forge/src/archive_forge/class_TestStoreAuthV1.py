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
class TestStoreAuthV1(base.MultiStoreBaseTest, SwiftTests, test_store_capabilities.TestStoreCapabilitiesChecking):
    _CONF = cfg.ConfigOpts()

    def getConfig(self):
        conf = SWIFT_CONF.copy()
        conf['swift_store_auth_version'] = '1'
        conf['swift_store_user'] = 'tenant:user1'
        return conf

    def setUp(self):
        """Establish a clean test environment."""
        super(TestStoreAuthV1, self).setUp()
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
        config = self.getConfig()
        conf_file = 'glance-swift.conf'
        self.swift_config_file = self.copy_data_file(conf_file, self.test_dir)
        config.update({'swift_store_config_file': self.swift_config_file})
        self.stub_out_swiftclient(config['swift_store_auth_version'])
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.config(group='swift1', **config)
        self.store.configure()
        self.register_store_backend_schemes(self.store, 'swift', 'swift1')
        self.addCleanup(self.conf.reset)