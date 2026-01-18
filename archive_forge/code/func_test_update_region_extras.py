import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_update_region_extras(self):
    new_region = unit.new_region_ref()
    region_id = new_region['id']
    PROVIDERS.catalog_api.create_region(new_region)
    email = 'keystone@openstack.org'
    new_ref = {'description': uuid.uuid4().hex, 'email': email}
    PROVIDERS.catalog_api.update_region(region_id, new_ref)
    current_region = PROVIDERS.catalog_api.get_region(region_id)
    self.assertEqual(email, current_region['email'])