import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_project_group_role_grant_inherited(self):
    group_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_url('PUT', ['OS-INHERIT', 'projects', project_id, 'groups', group_id, self.collection_key, ref['id'], 'inherited_to_projects'], status_code=204)
    self.manager.grant(role=ref['id'], project=project_id, group=group_id, os_inherit_extension_inherited=True)