import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def _set_auth_info(self) -> None:
    tokens = [token for token in [self.private_token, self.oauth_token, self.job_token] if token]
    if len(tokens) > 1:
        raise ValueError('Only one of private_token, oauth_token or job_token should be defined')
    if self.http_username and (not self.http_password) or (not self.http_username and self.http_password):
        raise ValueError('Both http_username and http_password should be defined')
    if tokens and self.http_username:
        raise ValueError('Only one of token authentications or http authentication should be defined')
    self._auth: Optional[requests.auth.AuthBase] = None
    if self.private_token:
        self._auth = _backends.PrivateTokenAuth(self.private_token)
    if self.oauth_token:
        self._auth = _backends.OAuthTokenAuth(self.oauth_token)
    if self.job_token:
        self._auth = _backends.JobTokenAuth(self.job_token)
    if self.http_username and self.http_password:
        self._auth = requests.auth.HTTPBasicAuth(self.http_username, self.http_password)