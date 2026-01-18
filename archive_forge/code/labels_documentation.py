from typing import Any, cast, Dict, Optional, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
Update a Label on the server.

        Args:
            name: The name of the label
            **kwargs: Extra options to send to the server (e.g. sudo)
        