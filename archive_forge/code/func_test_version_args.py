import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_version_args(self):
    """Validate _normalize_version_args."""

    def assert_min_max(in_ver, in_min, in_max, in_type, out_min, out_max):
        self.assertEqual((out_min, out_max), discover._normalize_version_args(in_ver, in_min, in_max, service_type=in_type))

    def normalize_raises(ver, min, max, in_type):
        self.assertRaises(ValueError, discover._normalize_version_args, ver, min, max, service_type=in_type)
    assert_min_max(None, None, None, None, None, None)
    assert_min_max(None, None, 'v1.2', None, None, (1, 2))
    assert_min_max(None, 'v1.2', 'latest', None, (1, 2), (discover.LATEST, discover.LATEST))
    assert_min_max(None, 'v1.2', '1.6', None, (1, 2), (1, 6))
    assert_min_max(None, 'v1.2', '1.latest', None, (1, 2), (1, discover.LATEST))
    assert_min_max(None, 'latest', 'latest', None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
    assert_min_max(None, 'latest', None, None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
    assert_min_max(None, (1, 2), None, None, (1, 2), (discover.LATEST, discover.LATEST))
    assert_min_max('', ('1', '2'), (1, 6), None, (1, 2), (1, 6))
    assert_min_max(None, ('1', '2'), (1, discover.LATEST), None, (1, 2), (1, discover.LATEST))
    assert_min_max('v1.2', '', None, None, (1, 2), (1, discover.LATEST))
    assert_min_max('1.latest', None, '', None, (1, discover.LATEST), (1, discover.LATEST))
    assert_min_max('v1', None, None, None, (1, 0), (1, discover.LATEST))
    assert_min_max('latest', None, None, None, (discover.LATEST, discover.LATEST), (discover.LATEST, discover.LATEST))
    assert_min_max(None, None, 'latest', 'volumev2', (2, 0), (2, discover.LATEST))
    assert_min_max(None, None, None, 'volumev2', (2, 0), (2, discover.LATEST))
    normalize_raises('v1', 'v2', None, None)
    normalize_raises('v1', None, 'v2', None)
    normalize_raises(None, 'latest', 'v1', None)
    normalize_raises(None, 'v1.2', 'v1.1', None)
    normalize_raises(None, 'v1.2', 1, None)
    normalize_raises('v2', None, None, 'volumev3')
    normalize_raises('v3', None, None, 'volumev2')
    normalize_raises(None, 'v2', None, 'volumev3')
    normalize_raises(None, None, 'v3', 'volumev2')