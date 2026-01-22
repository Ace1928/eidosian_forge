from openstack import exceptions
from openstack import resource
class ResourceProviderInventory(resource.Resource):
    resource_key = None
    resources_key = None
    base_path = '/resource_providers/%(resource_provider_id)s/inventories'
    _query_mapping = resource.QueryParameters(include_pagination_defaults=False)
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    resource_provider_id = resource.URI('resource_provider_id')
    resource_class = resource.Body('resource_class', alternate_id=True)
    resource_provider_generation = resource.Body('resource_provider_generation', type=int)
    allocation_ratio = resource.Body('allocation_ratio', type=float)
    max_unit = resource.Body('max_unit', type=int)
    min_unit = resource.Body('min_unit', type=int)
    reserved = resource.Body('reserved', type=int)
    step_size = resource.Body('step_size', type=int)
    total = resource.Body('total', type=int)

    def commit(self, session, prepend_key=True, has_body=True, retry_on_conflict=None, base_path=None, *, microversion=None, **kwargs):
        self._body._dirty.add('resource_provider_generation')
        return super().commit(session, prepend_key=prepend_key, has_body=has_body, retry_on_conflict=retry_on_conflict, base_path=base_path, microversion=microversion, **kwargs)

    @classmethod
    def list(cls, session, paginated=True, base_path=None, allow_unknown_params=False, *, microversion=None, **params):
        """This method is a generator which yields resource objects.

        A re-implementation of :meth:`~openstack.resource.Resource.list` that
        handles placement's single, unpaginated list implementation.

        Refer to :meth:`~openstack.resource.Resource.list` for full
        documentation including parameter, exception and return type
        documentation.
        """
        session = cls._get_session(session)
        if microversion is None:
            microversion = cls._get_microversion(session, action='list')
        if base_path is None:
            base_path = cls.base_path
        client_filters = {}
        for k, v in params.items():
            if hasattr(cls, k) and isinstance(getattr(cls, k), resource.Body) and (k not in cls._query_mapping._mapping.keys()):
                client_filters[k] = v
        uri = base_path % params
        uri_params = {}
        for k, v in params.items():
            if hasattr(cls, k) and isinstance(getattr(cls, k), resource.URI):
                uri_params[k] = v

        def _dict_filter(f, d):
            """Dict param based filtering"""
            if not d:
                return False
            for key in f.keys():
                if isinstance(f[key], dict):
                    if not _dict_filter(f[key], d.get(key, None)):
                        return False
                elif d.get(key, None) != f[key]:
                    return False
            return True
        response = session.get(uri, headers={'Accept': 'application/json'}, params={}, microversion=microversion)
        exceptions.raise_from_response(response)
        data = response.json()
        for resource_class, resource_data in data['inventories'].items():
            resource_inventory = {'resource_class': resource_class, 'resource_provider_generation': data['resource_provider_generation'], **resource_data, **uri_params}
            value = cls.existing(microversion=microversion, connection=session._get_connection(), **resource_inventory)
            filters_matched = True
            for key in client_filters.keys():
                if isinstance(client_filters[key], dict):
                    if not _dict_filter(client_filters[key], value.get(key, None)):
                        filters_matched = False
                        break
                elif value.get(key, None) != client_filters[key]:
                    filters_matched = False
                    break
            if filters_matched:
                yield value
        return None