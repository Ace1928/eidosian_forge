import os_service_types
from os_service_types.tests import base
def test_singleton_same(self):
    service_types = os_service_types.get_service_types()
    self.assertTrue(service_types is self.service_types)