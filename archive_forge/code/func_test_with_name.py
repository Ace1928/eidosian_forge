import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_with_name(self):
    name = uuid.uuid4().hex
    secret = uuid.uuid4().hex
    username = uuid.uuid4().hex
    user_domain_id = uuid.uuid4().hex
    app_cred = self.create(application_credential_name=name, application_credential_secret=secret, username=username, user_domain_id=user_domain_id)
    ac_method = app_cred.auth_methods[0]
    self.assertEqual(name, ac_method.application_credential_name)
    self.assertEqual(secret, ac_method.application_credential_secret)
    self.assertEqual(username, ac_method.username)
    self.assertEqual(user_domain_id, ac_method.user_domain_id)