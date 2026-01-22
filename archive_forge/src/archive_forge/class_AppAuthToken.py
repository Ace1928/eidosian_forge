import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
class AppAuthToken(JWT):
    """
    This class is used to authenticate as a GitHub App with a single constant JWT.
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app
    """

    def __init__(self, token: str):
        assert isinstance(token, str)
        assert len(token) > 0
        self._token = token

    @property
    def token(self) -> str:
        return self._token