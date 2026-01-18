import os_service_types
from os_service_types.tests import base
def test_singleton_different(self):
    service_types = os_service_types.ServiceTypes()
    self.assertFalse(service_types is self.service_types)