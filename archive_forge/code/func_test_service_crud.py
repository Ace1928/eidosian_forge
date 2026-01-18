import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_service_crud(self):
    new_service = unit.new_service_ref()
    service_id = new_service['id']
    res = PROVIDERS.catalog_api.create_service(service_id, new_service)
    self.assertDictEqual(new_service, res)
    services = PROVIDERS.catalog_api.list_services()
    self.assertIn(service_id, [x['id'] for x in services])
    service_name_update = {'name': uuid.uuid4().hex}
    res = PROVIDERS.catalog_api.update_service(service_id, service_name_update)
    expected_service = new_service.copy()
    expected_service['name'] = service_name_update['name']
    self.assertDictEqual(expected_service, res)
    PROVIDERS.catalog_api.delete_service(service_id)
    self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, service_id)
    self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.get_service, service_id)