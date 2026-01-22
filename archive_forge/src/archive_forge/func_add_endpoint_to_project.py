import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def add_endpoint_to_project(self, endpoint_id, project_id):
    """Create an endpoint to project association.

        :param endpoint_id: identity of endpoint to associate
        :type endpoint_id: string
        :param project_id: identity of the project to be associated with
        :type project_id: string
        :raises: keystone.exception.Conflict: If the endpoint was already
            added to project.
        :returns: None.

        """
    raise exception.NotImplemented()