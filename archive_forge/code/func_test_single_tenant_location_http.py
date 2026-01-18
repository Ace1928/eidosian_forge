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