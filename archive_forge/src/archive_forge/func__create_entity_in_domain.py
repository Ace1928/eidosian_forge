from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _create_entity_in_domain(entity_type, domain_id):
    """Create a user or group entity in the domain."""
    if entity_type == 'users':
        new_entity = unit.new_user_ref(domain_id=domain_id)
        new_entity = PROVIDERS.identity_api.create_user(new_entity)
    elif entity_type == 'groups':
        new_entity = unit.new_group_ref(domain_id=domain_id)
        new_entity = PROVIDERS.identity_api.create_group(new_entity)
    elif entity_type == 'roles':
        new_entity = self._create_role(domain_id=domain_id)
    else:
        raise exception.NotImplemented()
    return new_entity