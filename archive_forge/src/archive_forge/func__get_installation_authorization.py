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
def _get_installation_authorization(self) -> InstallationAuthorization:
    assert self.__integration is not None, 'Method withRequester(Requester) must be called first'
    return self.__integration.get_access_token(self._installation_id, permissions=self._token_permissions)