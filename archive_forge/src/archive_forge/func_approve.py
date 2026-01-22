from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
@cli.register_custom_action('User')
@exc.on_http_error(exc.GitlabUserApproveError)
def approve(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Approve a user creation request.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUserApproveError: If the user could not be activated

        Returns:
            The new object data (*not* a RESTObject)
        """
    path = f'/users/{self.encoded_id}/approve'
    return self.manager.gitlab.http_post(path, **kwargs)