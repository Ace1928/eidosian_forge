import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
class TesterFederationPlugin(v3.FederationBaseAuth):

    def get_unscoped_auth_ref(self, sess, **kwargs):
        resp = sess.post(self.federated_token_url, authenticated=False)
        return access.create(resp=resp)