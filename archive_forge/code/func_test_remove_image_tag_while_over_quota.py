import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_remove_image_tag_while_over_quota(self):
    self.config(image_tag_quota=1)
    self.image.tags.add('foo')
    self.assertEqual(1, len(self.image.tags))
    self.config(image_tag_quota=0)
    self.image.tags.remove('foo')
    self.assertEqual(0, len(self.image.tags))