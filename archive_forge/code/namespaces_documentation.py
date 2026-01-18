from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RetrieveMixin
from gitlab.utils import EncodedId
Get existence of a namespace by path.

        Args:
            namespace: The path to the namespace.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            Data on namespace existence returned from the server.
        