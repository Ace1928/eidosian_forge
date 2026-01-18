import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_secret(self):
    oauth2_endpoint = 'https://localhost/token'
    self.assertRaises(exceptions.OptionError, self.create, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=uuid.uuid4().hex)