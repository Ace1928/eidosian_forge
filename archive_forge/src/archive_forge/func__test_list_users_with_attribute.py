import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _test_list_users_with_attribute(self, filters, fed_dict):
    self._build_fed_resource()
    domain = self._get_domain_fixture()
    hints = driver_hints.Hints()
    hints = self._build_hints(hints, filters, fed_dict)
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(0, len(users))
    hints = self._build_hints(hints, filters, fed_dict)
    PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict)
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))
    hints = self._build_hints(hints, filters, fed_dict)
    fed_dict2 = unit.new_federated_user_ref()
    fed_dict2['idp_id'] = 'myidp'
    fed_dict2['protocol_id'] = 'mapped'
    PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict2)
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))
    hints = self._build_hints(hints, filters, fed_dict)
    if not any(('unique_id' in x['name'] for x in hints.filters)):
        hints = self._build_hints(hints, filters, fed_dict)
        fed_dict3 = unit.new_federated_user_ref()
        for filters_ in hints.filters:
            if filters_['name'] == 'idp_id':
                fed_dict3['idp_id'] = fed_dict['idp_id']
            elif filters_['name'] == 'protocol_id':
                fed_dict3['protocol_id'] = fed_dict['protocol_id']
        PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict3)
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(2, len(users))