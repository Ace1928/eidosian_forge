import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def add_role_to_user_and_project(self, user_id, project_id, role_id):
    """Add a role to a user within given project.

        :raises keystone.exception.Conflict: If a duplicate role assignment
            exists.

        """
    raise exception.NotImplemented()