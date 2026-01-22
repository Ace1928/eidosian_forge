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
def _refresh(self) -> None:
    if self._refresh_token is None:
        raise RuntimeError('Cannot refresh expired token because no refresh token has been provided')
    if self._refresh_expires_at is not None and self._refresh_expires_at < datetime.now(timezone.utc):
        raise RuntimeError('Cannot refresh expired token because refresh token also expired')
    token = self.__app.refresh_access_token(self._refresh_token)
    self._token = token.token
    self._type = token.type
    self._scope = token.scope
    self._expires_at = token.expires_at
    self._refresh_token = token.refresh_token
    self._refresh_expires_at = token.refresh_expires_at