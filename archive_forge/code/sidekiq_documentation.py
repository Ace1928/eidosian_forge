from typing import Any, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager
Return all available metrics and statistics.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the information couldn't be retrieved

        Returns:
            All available Sidekiq metrics and statistics
        