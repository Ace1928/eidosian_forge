import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_project(self, project_id, project):
    """Update an existing project.

        :raises keystone.exception.ProjectNotFound: if project_id does not
                                                    exist
        :raises keystone.exception.Conflict: if project name already exists

        """
    raise exception.NotImplemented()