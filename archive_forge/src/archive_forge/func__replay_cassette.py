from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
def _replay_cassette(self):
    plugin = v2.Password(auth_url=self.TEST_AUTH_URL, password=self.TEST_PASSWORD, username=self.TEST_USERNAME, tenant_name=self.TEST_TENANT_NAME)
    s = session.Session()
    s.get_token(auth=plugin)