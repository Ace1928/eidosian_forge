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
def test_read_acl_public(self):
    """
        Test that we can set a public read acl.
        """
    self.config(group='swift1', swift_store_config_file=None)
    self.config(group='swift1', swift_store_multi_tenant=True)
    store = Store(self.conf, backend='swift1')
    store.configure()
    uri = 'swift+http://storeurl/glance/%s' % FAKE_UUID
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    ctxt = mock.MagicMock()
    store.set_acls(loc, public=True, context=ctxt)
    container_headers = swiftclient.client.head_container('x', 'y', 'glance')
    self.assertEqual('*:*', container_headers['X-Container-Read'])