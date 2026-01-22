from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
class TestProtectedImageRepoProxy(utils.BaseTestCase):

    class ImageRepoStub(object):

        def __init__(self, fixtures):
            self.fixtures = fixtures

        def get(self, image_id):
            for f in self.fixtures:
                if f.image_id == image_id:
                    return f
            else:
                raise ValueError(image_id)

        def list(self, *args, **kwargs):
            return self.fixtures

        def add(self, image):
            self.fixtures.append(image)

    def setUp(self):
        super(TestProtectedImageRepoProxy, self).setUp()
        self.set_property_protections()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)
        self.property_rules = property_utils.PropertyRules(self.policy)
        self.image_factory = glance.domain.ImageFactory()
        extra_props = {'spl_create_prop': 'c', 'spl_read_prop': 'r', 'spl_update_prop': 'u', 'spl_delete_prop': 'd', 'forbidden': 'prop'}
        extra_props_2 = {'spl_read_prop': 'r', 'forbidden': 'prop'}
        self.fixtures = [self.image_factory.new_image(image_id='1', owner=TENANT1, extra_properties=extra_props), self.image_factory.new_image(owner=TENANT2, visibility='public'), self.image_factory.new_image(image_id='3', owner=TENANT1, extra_properties=extra_props_2)]
        self.context = glance.context.RequestContext(roles=['spl_role'])
        image_repo = self.ImageRepoStub(self.fixtures)
        self.image_repo = property_protections.ProtectedImageRepoProxy(image_repo, self.context, self.property_rules)

    def test_get_image(self):
        image_id = '1'
        result_image = self.image_repo.get(image_id)
        result_extra_props = result_image.extra_properties
        self.assertEqual('c', result_extra_props['spl_create_prop'])
        self.assertEqual('r', result_extra_props['spl_read_prop'])
        self.assertEqual('u', result_extra_props['spl_update_prop'])
        self.assertEqual('d', result_extra_props['spl_delete_prop'])
        self.assertNotIn('forbidden', result_extra_props.keys())

    def test_list_image(self):
        result_images = self.image_repo.list()
        self.assertEqual(3, len(result_images))
        result_extra_props = result_images[0].extra_properties
        self.assertEqual('c', result_extra_props['spl_create_prop'])
        self.assertEqual('r', result_extra_props['spl_read_prop'])
        self.assertEqual('u', result_extra_props['spl_update_prop'])
        self.assertEqual('d', result_extra_props['spl_delete_prop'])
        self.assertNotIn('forbidden', result_extra_props.keys())
        result_extra_props = result_images[1].extra_properties
        self.assertEqual({}, result_extra_props)
        result_extra_props = result_images[2].extra_properties
        self.assertEqual('r', result_extra_props['spl_read_prop'])
        self.assertNotIn('forbidden', result_extra_props.keys())