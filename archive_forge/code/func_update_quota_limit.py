import abc
@staticmethod
@abc.abstractmethod
def update_quota_limit(context, project_id, resource, limit):
    """Update the quota limit for a resource in a project

        :param context: The request context, for access checks.
        :param project_id: The ID of the project to update the quota.
        :param resource: the resource to update the quota.
        :param limit: new resource quota limit.
        """