import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_sp_duplicate_conflict_gives_name(self):
    sp = {'auth_url': uuid.uuid4().hex, 'enabled': True, 'description': uuid.uuid4().hex, 'sp_url': uuid.uuid4().hex, 'relay_state_prefix': CONF.saml.relay_state_prefix}
    service_ref = PROVIDERS.federation_api.create_sp('SP1', sp)
    try:
        PROVIDERS.federation_api.create_sp('SP1', sp)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with ID %s' % service_ref['id'], repr(e))
    else:
        self.fail('Create duplicate sp did not raise a conflict')