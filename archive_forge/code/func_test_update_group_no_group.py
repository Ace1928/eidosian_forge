import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_group_no_group(self):
    group_mod = {'description': uuid.uuid4().hex}
    self.assertRaises(exception.GroupNotFound, self.driver.update_group, group_id=uuid.uuid4().hex, group=group_mod)