import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def _build_role_assignment_response(self, role_id, scope_type, scope_id, entity_type, entity_id):
    self.assertThat(['group', 'user'], matchers.Contains(entity_type))
    self.assertThat(['project', 'domain'], matchers.Contains(scope_type))
    link_str = 'https://identity.example.com/identity/v3/{scope_t}s/{scopeid}/{entity_t}s/{entityid}/roles/{roleid}'
    return [{'links': {'assignment': link_str.format(scope_t=scope_type, scopeid=scope_id, entity_t=entity_type, entityid=entity_id, roleid=role_id)}, 'role': {'id': role_id}, 'scope': {scope_type: {'id': scope_id}}, entity_type: {'id': entity_id}}]