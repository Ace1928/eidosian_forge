import uuid
from osc_placement.tests.functional import base
class TestResourceClass17(base.BaseTestCase):
    VERSION = '1.7'

    def test_set_resource_class(self):
        self.resource_class_create(CUSTOM_RC)
        self.resource_class_set(CUSTOM_RC)
        self.resource_class_set(CUSTOM_RC + '1')
        rcs = self.resource_class_list()
        names = [rc['name'] for rc in rcs]
        self.assertIn(CUSTOM_RC, names)
        self.assertIn(CUSTOM_RC + '1', names)
        self.resource_class_delete(CUSTOM_RC)
        self.resource_class_delete(CUSTOM_RC + '1')