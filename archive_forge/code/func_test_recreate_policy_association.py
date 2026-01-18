import uuid
from keystone import exception
def test_recreate_policy_association(self):
    original_association = self.create_association(service_id=uuid.uuid4().hex)
    override_association = original_association.copy()
    override_association['policy_id'] = uuid.uuid4().hex
    self.driver.create_policy_association(**override_association)
    self.driver.check_policy_association(**override_association)
    self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **original_association)