import http.client as http
from unittest.mock import patch
from oslo_log.fixture import logging_error as log_fixture
from oslo_policy import policy
from oslo_utils.fixture import uuidsentinel as uuids
import testtools
import webob
import glance.api.middleware.cache
import glance.api.policy
from glance.common import exception
from glance import context
from glance.tests.unit import base
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import test_policy
from glance.tests.unit import utils as unit_test_utils
def test_verify_metadata_zero_size(self):
    """
        Test verify_metadata updates metadata with cached image size for images
        with 0 size
        """
    image_meta = {'size': 0, 'deleted': False, 'id': 'test1', 'status': 'active'}
    self._test_verify_metadata_zero_size(image_meta)