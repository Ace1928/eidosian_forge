from oslotest import base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
class TestResourceExtendClass(base.BaseTestCase):

    def test_extends(self):
        self.assertIsNotNone(resource_extend.get_funcs('ExtendedA'))
        self.assertIsNotNone(resource_extend.get_funcs('ExtendedB'))