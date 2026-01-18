from typing import Any, Callable, cast, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import FileAttribute, RequiredOptional
Download the secure file.

        Args:
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the artifacts could not be retrieved

        Returns:
            The artifacts if `streamed` is False, None otherwise.