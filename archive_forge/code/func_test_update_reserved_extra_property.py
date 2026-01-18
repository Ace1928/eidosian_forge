from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_update_reserved_extra_property(self):
    extra_properties = {'spl_create_prop': 'bar'}
    context = glance.context.RequestContext(roles=['spl_role'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    self.assertRaises(exception.ReservedProperty, extra_prop_proxy.__setitem__, 'spl_create_prop', 'par')