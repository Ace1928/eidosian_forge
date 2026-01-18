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
def test_with_v3_auth(self):
    self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'auth_version': '3', 'user': 'user:pass', 'key': 'password', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}}
    self.store.configure()
    connection = self.store.get_connection(self.location)
    self.assertEqual('3', connection.auth_version)
    self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}, connection.os_options)