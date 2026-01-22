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
def _create_fake_image(self, context, size):
    location_count = 2
    locations = []
    for i in range(location_count):
        locations.append({'url': 'file:///g/there/it/is%d' % i, 'status': 'active', 'metadata': {}})
    image_values = {'id': str(uuid.uuid4()), 'owner': context.owner, 'status': 'active', 'size': size * units.Mi, 'locations': locations}
    self.db_api.image_create(context, image_values)