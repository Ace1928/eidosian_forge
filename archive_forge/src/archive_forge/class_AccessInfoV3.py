import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
class AccessInfoV3(AccessInfo):
    """An object encapsulating raw v3 auth token from identity service."""
    version = 'v3'
    _service_catalog_class = service_catalog.ServiceCatalogV3

    def has_service_catalog(self):
        return 'catalog' in self._data['token']

    @property
    def _user(self):
        return self._data['token']['user']

    @property
    def is_federated(self):
        return 'OS-FEDERATION' in self._user

    @property
    def is_admin_project(self):
        return self._data.get('token', {}).get('is_admin_project', True)

    @_missingproperty
    def expires(self):
        return utils.parse_isotime(self._data['token']['expires_at'])

    @_missingproperty
    def issued(self):
        return utils.parse_isotime(self._data['token']['issued_at'])

    @_missingproperty
    def user_id(self):
        return self._user['id']

    @property
    def user_domain_id(self):
        try:
            return self._user['domain']['id']
        except KeyError:
            if self.is_federated:
                return None
            raise

    @property
    def user_domain_name(self):
        try:
            return self._user['domain']['name']
        except KeyError:
            if self.is_federated:
                return None
            raise

    @_missingproperty
    def role_ids(self):
        return [r['id'] for r in self._data['token'].get('roles', [])]

    @_missingproperty
    def role_names(self):
        return [r['name'] for r in self._data['token'].get('roles', [])]

    @_missingproperty
    def username(self):
        return self._user['name']

    @_missingproperty
    def system(self):
        return self._data['token']['system']

    @property
    def _domain(self):
        return self._data['token']['domain']

    @_missingproperty
    def domain_name(self):
        return self._domain['name']

    @_missingproperty
    def domain_id(self):
        return self._domain['id']

    @property
    def _project(self):
        return self._data['token']['project']

    @_missingproperty
    def project_id(self):
        return self._project['id']

    @_missingproperty
    def project_is_domain(self):
        return self._data['token']['is_domain']

    @_missingproperty
    def project_domain_id(self):
        return self._project['domain']['id']

    @_missingproperty
    def project_domain_name(self):
        return self._project['domain']['name']

    @_missingproperty
    def project_name(self):
        return self._project['name']

    @property
    def domain_scoped(self):
        try:
            return bool(self._domain)
        except KeyError:
            return False

    @_missingproperty
    def system_scoped(self):
        return self._data['token']['system'].get('all', False)

    @property
    def _trust(self):
        return self._data['token']['OS-TRUST:trust']

    @_missingproperty
    def trust_id(self):
        return self._trust['id']

    @property
    def trust_scoped(self):
        try:
            return bool(self._trust)
        except KeyError:
            return False

    @_missingproperty
    def trustee_user_id(self):
        return self._trust['trustee_user']['id']

    @_missingproperty
    def trustor_user_id(self):
        return self._trust['trustor_user']['id']

    @property
    def application_credential(self):
        return self._data['token']['application_credential']

    @_missingproperty
    def application_credential_id(self):
        return self._data['token']['application_credential']['id']

    @_missingproperty
    def application_credential_access_rules(self):
        return self._data['token']['application_credential']['access_rules']

    @property
    def _oauth(self):
        return self._data['token']['OS-OAUTH1']

    @_missingproperty
    def oauth_access_token_id(self):
        return self._oauth['access_token_id']

    @_missingproperty
    def oauth_consumer_id(self):
        return self._oauth['consumer_id']

    @_missingproperty
    def audit_id(self):
        try:
            return self._data['token']['audit_ids'][0]
        except IndexError:
            return None

    @_missingproperty
    def audit_chain_id(self):
        try:
            return self._data['token']['audit_ids'][1]
        except IndexError:
            return None

    @property
    def service_providers(self):
        if not self._service_providers:
            self._service_providers = service_providers.ServiceProviders.from_token(self._data)
        return self._service_providers

    @_missingproperty
    def bind(self):
        return self._data['token']['bind']

    @property
    def oauth2_credential(self):
        return self._data['token']['oauth2_credential']

    @_missingproperty
    def oauth2_credential_thumbprint(self):
        return self._data['token']['oauth2_credential']['x5t#S256']