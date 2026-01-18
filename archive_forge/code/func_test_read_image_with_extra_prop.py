from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_read_image_with_extra_prop(self):
    context = glance.context.RequestContext(roles=['spl_role'])
    extra_prop = {'spl_read_prop': 'read', 'spl_fake_prop': 'prop'}
    image = self.ImageStub(extra_prop)
    result_image = property_protections.ProtectedImageProxy(image, context, self.property_rules)
    result_extra_props = result_image.extra_properties
    self.assertEqual('read', result_extra_props['spl_read_prop'])
    self.assertNotIn('spl_fake_prop', result_extra_props.keys())