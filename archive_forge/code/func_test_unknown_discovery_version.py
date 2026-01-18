import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_unknown_discovery_version(self):
    self.stub_discovery(v2=False, v3_id='v4.0')
    self.assertDiscoveryFailure()