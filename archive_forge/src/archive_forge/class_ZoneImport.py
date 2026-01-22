from openstack.dns.v2 import _base
from openstack import exceptions
from openstack import resource
class ZoneImport(_base.Resource):
    """DNS Zone Import Resource"""
    resource_key = ''
    resources_key = 'imports'
    base_path = '/zones/tasks/import'
    allow_create = True
    allow_fetch = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('zone_id', 'message', 'status')
    created_at = resource.Body('created_at')
    links = resource.Body('links', type=dict)
    message = resource.Body('message')
    metadata = resource.Body('metadata', type=list)
    project_id = resource.Body('project_id')
    status = resource.Body('status')
    updated_at = resource.Body('updated_at')
    version = resource.Body('version', type=int)
    zone_id = resource.Body('zone_id')

    def create(self, session, prepend_key=True, base_path=None):
        """Create a remote resource based on this instance.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param prepend_key: A boolean indicating whether the resource_key
                            should be prepended in a resource creation
                            request. Default to True.
        :param str base_path: Base part of the URI for creating resources, if
                              different from
                              :data:`~openstack.resource.Resource.base_path`.
        :return: This :class:`Resource` instance.
        :raises: :exc:`~openstack.exceptions.MethodNotSupported` if
                 :data:`Resource.allow_create` is not set to ``True``.
        """
        if not self.allow_create:
            raise exceptions.MethodNotSupported(self, 'create')
        session = self._get_session(session)
        microversion = self._get_microversion(session, action='create')
        request = resource._Request(self.base_path, None, {'content-type': 'text/dns'})
        response = session.post(request.url, json=request.body, headers=request.headers, microversion=microversion)
        self.microversion = microversion
        self._translate_response(response)
        return self