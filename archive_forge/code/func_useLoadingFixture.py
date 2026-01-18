import uuid
from oslo_config import fixture as config
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def useLoadingFixture(self, **kwargs):
    return self.useFixture(fixture.LoadingFixture(**kwargs))