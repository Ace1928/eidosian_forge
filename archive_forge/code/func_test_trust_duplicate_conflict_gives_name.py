import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_trust_duplicate_conflict_gives_name(self):
    trustor = unit.new_user_ref(domain_id=self.domain_id)
    trustor = PROVIDERS.identity_api.create_user(trustor)
    trustee = unit.new_user_ref(domain_id=self.domain_id)
    trustee = PROVIDERS.identity_api.create_user(trustee)
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    trust_ref = unit.new_trust_ref(trustor['id'], trustee['id'])
    PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, [role_ref])
    try:
        PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, [role_ref])
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with ID %s' % trust_ref['id'], repr(e))
    else:
        self.fail('Create duplicate trust did not raise a conflict')