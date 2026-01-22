import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.identity import access as access_plugin
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def _plugin(self, **kwargs):
    token = fixture.V3Token()
    s = token.add_service('identity')
    s.add_standard_endpoints(public=self.TEST_ROOT_URL)
    auth_ref = access.create(body=token, auth_token=self.auth_token)
    return access_plugin.AccessInfoPlugin(auth_ref, **kwargs)