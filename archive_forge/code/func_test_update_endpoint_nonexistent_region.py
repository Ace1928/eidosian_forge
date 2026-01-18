import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_update_endpoint_nonexistent_region(self):
    dummy_service, enabled_endpoint, dummy_disabled_endpoint = self._create_endpoints()
    new_endpoint = unit.new_endpoint_ref(service_id=uuid.uuid4().hex)
    self.assertRaises(exception.ValidationError, PROVIDERS.catalog_api.update_endpoint, enabled_endpoint['id'], new_endpoint)