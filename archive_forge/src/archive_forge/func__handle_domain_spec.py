from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _handle_domain_spec(self, test_data, domain_spec):
    """Handle the creation of domains and their contents.

        domain_spec may either be a count of the number of empty domains to
        create, a dict describing the domain contents, or a list of
        domain_specs.

        In the case when a list is provided, this method calls itself
        recursively to handle the list elements.

        This method will insert any entities created into test_data

        """

    def _create_domain(domain_id=None):
        if domain_id is None:
            new_domain = unit.new_domain_ref()
            PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
            return new_domain
        else:
            return PROVIDERS.resource_api.get_domain(domain_id)

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
    if isinstance(domain_spec, list):
        for x in domain_spec:
            self._handle_domain_spec(test_data, x)
    elif isinstance(domain_spec, dict):
        the_domain = _create_domain(domain_spec.get('id'))
        test_data['domains'].append(the_domain)
        for entity_type, value in domain_spec.items():
            if entity_type == 'id':
                continue
            if entity_type == 'projects':
                self._handle_project_spec(test_data, the_domain['id'], value)
            else:
                for _ in range(value):
                    test_data[entity_type].append(_create_entity_in_domain(entity_type, the_domain['id']))
    else:
        for _ in range(domain_spec):
            test_data['domains'].append(_create_domain())