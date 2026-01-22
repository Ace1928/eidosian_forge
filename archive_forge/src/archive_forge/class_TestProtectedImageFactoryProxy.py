from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
class TestProtectedImageFactoryProxy(utils.BaseTestCase):

    def setUp(self):
        super(TestProtectedImageFactoryProxy, self).setUp()
        self.set_property_protections()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)
        self.property_rules = property_utils.PropertyRules(self.policy)
        self.factory = glance.domain.ImageFactory()

    def test_create_image_no_extra_prop(self):
        self.context = glance.context.RequestContext(tenant=TENANT1, roles=['spl_role'])
        self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
        extra_props = {}
        image = self.image_factory.new_image(extra_properties=extra_props)
        expected_extra_props = {}
        self.assertEqual(expected_extra_props, image.extra_properties)

    def test_create_image_extra_prop(self):
        self.context = glance.context.RequestContext(tenant=TENANT1, roles=['spl_role'])
        self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
        extra_props = {'spl_create_prop': 'c'}
        image = self.image_factory.new_image(extra_properties=extra_props)
        expected_extra_props = {'spl_create_prop': 'c'}
        self.assertEqual(expected_extra_props, image.extra_properties)

    def test_create_image_extra_prop_reserved_property(self):
        self.context = glance.context.RequestContext(tenant=TENANT1, roles=['spl_role'])
        self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
        extra_props = {'foo': 'bar', 'spl_create_prop': 'c'}
        self.assertRaises(exception.ReservedProperty, self.image_factory.new_image, extra_properties=extra_props)

    def test_create_image_extra_prop_admin(self):
        self.context = glance.context.RequestContext(tenant=TENANT1, roles=['admin'])
        self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
        extra_props = {'foo': 'bar', 'spl_create_prop': 'c'}
        image = self.image_factory.new_image(extra_properties=extra_props)
        expected_extra_props = {'foo': 'bar', 'spl_create_prop': 'c'}
        self.assertEqual(expected_extra_props, image.extra_properties)

    def test_create_image_extra_prop_invalid_role(self):
        self.context = glance.context.RequestContext(tenant=TENANT1, roles=['imaginary-role'])
        self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
        extra_props = {'foo': 'bar', 'spl_create_prop': 'c'}
        self.assertRaises(exception.ReservedProperty, self.image_factory.new_image, extra_properties=extra_props)