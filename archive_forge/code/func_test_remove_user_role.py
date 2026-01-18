import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
def test_remove_user_role(self):
    self.stub_url('DELETE', ['users', 'foo', 'roles', 'OS-KSADM', 'barrr'], status_code=204)
    self.client.roles.remove_user_role('foo', 'barrr')