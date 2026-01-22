import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _list_users_in_group_data(self):
    number_of_users = 10
    user_name_data = {1: 'Arthur Conan Doyle', 3: 'Arthur Rimbaud', 9: 'Arthur Schopenhauer'}
    user_list = self._create_test_data('user', number_of_users, domain_id=CONF.identity.default_domain_id, name_dict=user_name_data)
    group = self._create_one_entity('group', CONF.identity.default_domain_id, 'Great Writers')
    for i in range(7):
        PROVIDERS.identity_api.add_user_to_group(user_list[i]['id'], group['id'])
    return (user_list, group)