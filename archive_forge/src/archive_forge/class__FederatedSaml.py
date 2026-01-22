import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
class _FederatedSaml(v3.FederationBaseAuth):

    def __init__(self, auth_url, identity_provider, protocol, identity_provider_url, **kwargs):
        super(_FederatedSaml, self).__init__(auth_url, identity_provider, protocol, **kwargs)
        self.identity_provider_url = identity_provider_url

    @abc.abstractmethod
    def get_requests_auth(self):
        raise NotImplementedError()

    def get_unscoped_auth_ref(self, session, **kwargs):
        method = self.get_requests_auth()
        auth = _SamlAuth(self.identity_provider_url, method)
        try:
            resp = session.get(self.federated_token_url, requests_auth=auth, authenticated=False)
        except SamlException as e:
            raise exceptions.AuthorizationFailure(str(e))
        return access.create(resp=resp)