import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
class AccessInfoV2(AccessInfo):
    """An object for encapsulating raw v2 auth token from identity service."""
    version = 'v2.0'
    _service_catalog_class = service_catalog.ServiceCatalogV2

    def has_service_catalog(self):
        return 'serviceCatalog' in self._data.get('access', {})

    @_missingproperty
    def auth_token(self):
        set_token = super(AccessInfoV2, self).auth_token
        return set_token or self._data['access']['token']['id']

    @property
    def _token(self):
        return self._data['access']['token']

    @_missingproperty
    def expires(self):
        return utils.parse_isotime(self._token.get('expires'))

    @_missingproperty
    def issued(self):
        return utils.parse_isotime(self._token['issued_at'])

    @property
    def _user(self):
        return self._data['access']['user']

    @_missingproperty
    def username(self):
        return self._user.get('name') or self._user.get('username')

    @_missingproperty
    def user_id(self):
        return self._user['id']

    @property
    def user_domain_id(self):
        return None

    @property
    def user_domain_name(self):
        return None

    @_missingproperty
    def role_ids(self):
        metadata = self._data.get('access', {}).get('metadata', {})
        return metadata.get('roles', [])

    @_missingproperty
    def role_names(self):
        return [r['name'] for r in self._user.get('roles', [])]

    @property
    def domain_name(self):
        return None

    @property
    def domain_id(self):
        return None

    @property
    def project_name(self):
        try:
            tenant_dict = self._token['tenant']
        except KeyError:
            pass
        else:
            return tenant_dict.get('name')
        try:
            return self._user['tenantName']
        except KeyError:
            pass
        try:
            return self._token['tenantId']
        except KeyError:
            pass

    @property
    def domain_scoped(self):
        return False

    @property
    def system_scoped(self):
        return False

    @property
    def _trust(self):
        return self._data['access']['trust']

    @_missingproperty
    def trust_id(self):
        return self._trust['id']

    @_missingproperty
    def trust_scoped(self):
        return bool(self._trust)

    @_missingproperty
    def trustee_user_id(self):
        return self._trust['trustee_user_id']

    @property
    def trustor_user_id(self):
        return None

    @property
    def project_id(self):
        try:
            tenant_dict = self._token['tenant']
        except KeyError:
            pass
        else:
            return tenant_dict.get('id')
        try:
            return self._user['tenantId']
        except KeyError:
            pass
        try:
            return self._token['tenantId']
        except KeyError:
            pass

    @property
    def project_is_domain(self):
        return False

    @property
    def project_domain_id(self):
        return None

    @property
    def project_domain_name(self):
        return None

    @property
    def oauth_access_token_id(self):
        return None

    @property
    def oauth_consumer_id(self):
        return None

    @property
    def is_federated(self):
        return False

    @property
    def is_admin_project(self):
        return True

    @property
    def audit_id(self):
        try:
            return self._token.get('audit_ids', [])[0]
        except IndexError:
            return None

    @property
    def audit_chain_id(self):
        try:
            return self._token.get('audit_ids', [])[1]
        except IndexError:
            return None

    @property
    def service_providers(self):
        return None

    @_missingproperty
    def bind(self):
        return self._token['bind']