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
def test_verify_metadata_is_image_target_instance_with_zero_size(self):
    """
        Test verify_metadata updates metadata which is ImageTarget instance
        """
    image = ImageStub('test1', uuids.owner)
    image.size = 0
    image_meta = glance.api.policy.ImageTarget(image)
    self._test_verify_metadata_zero_size(image_meta)