import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_user_domain(self):
    self.assertRaises(exceptions.OptionError, self.create, application_credential_name=uuid.uuid4().hex, username=uuid.uuid4().hex, application_credential_secret=uuid.uuid4().hex)