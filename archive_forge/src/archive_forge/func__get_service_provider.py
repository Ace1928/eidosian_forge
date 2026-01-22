from keystoneauth1 import exceptions
def _get_service_provider(self, sp_id):
    try:
        return self._service_providers[sp_id]
    except KeyError:
        raise exceptions.ServiceProviderNotFound(sp_id)