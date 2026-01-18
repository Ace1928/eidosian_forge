import copy
import os_service_types.data
from os_service_types import exc
Return the information for every service associated with a project.

        :param name: A repository or project name in the form
            ``'openstack/{project}'`` or just ``'{project}'``.
        :type name: str
        :raises ValueError: If project_name is None
        :returns: list of dicts
        