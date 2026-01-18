from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_save_tag(self):
    tag = self.tag_repo.get(NAMESPACE1, TAG1)
    self.tag_repo.save(tag)
    tag = self.tag_repo.get(NAMESPACE1, TAG1)
    self.assertEqual(TAG1, tag.name)