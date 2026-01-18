from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_update_doesnt_log_password(self):
    password = uuid.uuid4().hex
    ref = self.new_ref()
    req_ref = ref.copy()
    req_ref.pop('id')
    param_ref = req_ref.copy()
    self.stub_entity('PATCH', [self.collection_key, ref['id']], status_code=200, entity=ref)
    param_ref['password'] = password
    params = utils.parameterize(param_ref)
    self.manager.update(ref['id'], **params)
    self.assertNotIn(password, self.logger.output)