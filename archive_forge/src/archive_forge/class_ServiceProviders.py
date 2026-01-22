from keystoneauth1 import exceptions
class ServiceProviders(object):
    """Helper methods for dealing with Service Providers."""

    @classmethod
    def from_token(cls, token):
        if 'token' not in token:
            raise ValueError('Token format does not support serviceproviders.')
        return cls(token['token'].get('service_providers', []))

    def __init__(self, service_providers):

        def normalize(service_providers_list):
            return dict(((sp['id'], sp) for sp in service_providers_list if 'id' in sp))
        self._service_providers = normalize(service_providers)

    def _get_service_provider(self, sp_id):
        try:
            return self._service_providers[sp_id]
        except KeyError:
            raise exceptions.ServiceProviderNotFound(sp_id)

    def get_sp_url(self, sp_id):
        return self._get_service_provider(sp_id).get('sp_url')

    def get_auth_url(self, sp_id):
        return self._get_service_provider(sp_id).get('auth_url')