from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
Trigger a new scheduled pipeline, which runs immediately.
        The next scheduled run of this pipeline is not affected.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPipelinePlayError: If the request failed
        