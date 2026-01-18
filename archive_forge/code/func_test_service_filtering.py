import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_service_filtering(self):
    target_service = self._create_random_service()
    unrelated_service1 = self._create_random_service()
    unrelated_service2 = self._create_random_service()
    hint_for_type = driver_hints.Hints()
    hint_for_type.add_filter(name='type', value=target_service['type'])
    services = PROVIDERS.catalog_api.list_services(hint_for_type)
    self.assertEqual(1, len(services))
    filtered_service = services[0]
    self.assertEqual(target_service['type'], filtered_service['type'])
    self.assertEqual(target_service['id'], filtered_service['id'])
    self.assertEqual(0, len(hint_for_type.filters))
    hint_for_name = driver_hints.Hints()
    hint_for_name.add_filter(name='name', value=target_service['name'])
    services = PROVIDERS.catalog_api.list_services(hint_for_name)
    self.assertEqual(3, len(services))
    self.assertEqual(1, len(hint_for_name.filters))
    PROVIDERS.catalog_api.delete_service(target_service['id'])
    PROVIDERS.catalog_api.delete_service(unrelated_service1['id'])
    PROVIDERS.catalog_api.delete_service(unrelated_service2['id'])