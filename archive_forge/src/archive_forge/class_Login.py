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
class Login(HTTPBasicAuth):
    """
    This class is used to authenticate with login and password.
    """

    def __init__(self, login: str, password: str):
        assert isinstance(login, str)
        assert len(login) > 0
        assert isinstance(password, str)
        assert len(password) > 0
        self._login = login
        self._password = password

    @property
    def login(self) -> str:
        return self._login

    @property
    def username(self) -> str:
        return self.login

    @property
    def password(self) -> str:
        return self._password