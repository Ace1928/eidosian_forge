import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_update_registered_limit_duplicate(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
    update_ref = {'id': registered_limit_1['id'], 'region_id': self.region_two['id'], 'resource_name': 'snapshot'}
    self.assertRaises(exception.Conflict, PROVIDERS.unified_limit_api.update_registered_limit, registered_limit_1['id'], update_ref)