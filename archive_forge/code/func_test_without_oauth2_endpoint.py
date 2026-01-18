import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_without_oauth2_endpoint(self):
    client_id = uuid.uuid4().hex
    self.assertRaises(exceptions.OptionError, self.create, oauth2_client_id=client_id)