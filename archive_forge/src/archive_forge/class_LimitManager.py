from keystoneclient import base
class LimitManager(base.CrudManager):
    """Manager class for project limits."""
    resource_class = Limit
    collection_key = 'limits'
    key = 'limit'

    def create(self, project, service, resource_name, resource_limit, description=None, region=None, **kwargs):
        """Create a project-specific limit.

        :param project: the project to create a limit for.
        :type project: str or :class:`keystoneclient.v3.projects.Project`
        :param service: the service that owns the resource to limit.
        :type service: str or :class:`keystoneclient.v3.services.Service`
        :param resource_name: the name of the resource to limit
        :type resource_name: str
        :param resource_limit: the quantity of the limit
        :type resource_limit: int
        :param description: a description of the limit
        :type description: str
        :param region: region the limit applies to
        :type region: str or :class:`keystoneclient.v3.regions.Region`

        :returns: a reference of the created limit
        :rtype: :class:`keystoneclient.v3.limits.Limit`

        """
        limit_data = base.filter_none(project_id=base.getid(project), service_id=base.getid(service), resource_name=resource_name, resource_limit=resource_limit, description=description, region_id=base.getid(region), **kwargs)
        body = {self.collection_key: [limit_data]}
        resp, body = self.client.post('/limits', body=body)
        limit = body[self.collection_key].pop()
        return self._prepare_return_value(resp, self.resource_class(self, limit))

    def update(self, limit, project=None, service=None, resource_name=None, resource_limit=None, description=None, **kwargs):
        """Update a project-specific limit.

        :param limit: a limit to update
        :param project: the project ID of the limit to update
        :type project: str or :class:`keystoneclient.v3.projects.Project`
        :param resource_limit: the limit of the limit's resource to update
        :type: resource_limit: int
        :param description: a description of the limit
        :type description: str

        :returns: a reference of the updated limit.
        :rtype: :class:`keystoneclient.v3.limits.Limit`

        """
        return super(LimitManager, self).update(limit_id=base.getid(limit), project_id=base.getid(project), service_id=base.getid(service), resource_name=resource_name, resource_limit=resource_limit, description=description, **kwargs)

    def get(self, limit):
        """Retrieve a project limit.

        :param limit:
            the project-specific limit to be retrieved.
        :type limit:
            str or :class:`keystoneclient.v3.limit.Limit`

        :returns: a project-specific limit
        :rtype: :class:`keystoneclient.v3.limit.Limit`

        """
        return super(LimitManager, self).get(limit_id=base.getid(limit))

    def list(self, service=None, region=None, resource_name=None, **kwargs):
        """List project-specific limits.

        Any parameter provided will be passed to the server as a filter

        :param service: service to filter limits by
        :type service: UUID or :class:`keystoneclient.v3.services.Service`
        :param region: region to filter limits by
        :type region: UUID or :class:`keystoneclient.v3.regions.Region`
        :param resource_name: the name of the resource to filter limits by
        :type resource_name: str

        :returns: a list of project-specific limits.
        :rtype: list of :class:`keystoneclient.v3.limits.Limit`

        """
        return super(LimitManager, self).list(service_id=base.getid(service), region_id=base.getid(region), resource_name=resource_name, **kwargs)

    def delete(self, limit):
        """Delete a project-specific limit.

        :param limit: the project-specific limit to be deleted.
        :type limit: str or :class:`keystoneclient.v3.limit.Limit`

        :returns: Response object with 204 status
        :rtype: :class:`requests.models.Response`

        """
        return super(LimitManager, self).delete(limit_id=base.getid(limit))