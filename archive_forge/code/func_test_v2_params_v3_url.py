import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_v2_params_v3_url(self):
    self.stub_discovery(v2=False)
    self.assertCreateV3()