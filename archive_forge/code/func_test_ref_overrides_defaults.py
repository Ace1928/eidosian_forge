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
def test_ref_overrides_defaults(self):
    self.config(group='swift1', swift_store_auth_version='2', swift_store_user='testuser', swift_store_key='testpass', swift_store_auth_address='testaddress', swift_store_endpoint_type='internalURL', swift_store_config_file='somefile')
    self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'auth_version': '3', 'user': 'user:pass', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}}
    self.store.configure()
    self.assertEqual('user:pass', self.store.user)
    self.assertEqual('3', self.store.auth_version)
    self.assertEqual('authurl.com', self.store.auth_address)
    self.assertEqual('default', self.store.user_domain_id)
    self.assertEqual('ignored', self.store.user_domain_name)
    self.assertEqual('default', self.store.project_domain_id)
    self.assertEqual('ignored', self.store.project_domain_name)