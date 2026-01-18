import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_group_name_already_exists(self):
    if not self.allows_name_update:
        self.skipTest("driver doesn't allow name update")
    domain_id = uuid.uuid4().hex
    group1 = self.create_group(domain_id=domain_id)
    group2 = self.create_group(domain_id=domain_id)
    group_mod = {'name': group1['name']}
    self.assertRaises(exception.Conflict, self.driver.update_group, group2['id'], group_mod)