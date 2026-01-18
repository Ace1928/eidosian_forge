from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_tag_not_found(self):
    fake_name = 'fake_name'
    tag = self.tag_repo.get(NAMESPACE1, TAG1)
    tag.name = fake_name
    self.assertRaises(exception.NotFound, self.tag_repo.remove, tag)