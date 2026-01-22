from typing import Any, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, ListMixin, ObjectDeleteMixin
Create or update the object.

        Args:
            name: The value to set for the object
            value: The value to set for the object
            feature_group: A feature group name
            user: A GitLab username
            group: A GitLab group
            project: A GitLab project in form group/project
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabSetError: If an error occurred

        Returns:
            The created/updated attribute
        