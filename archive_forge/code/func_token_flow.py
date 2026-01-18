from __future__ import annotations
import socket
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from lazyops.utils.helpers import timed_cache
from lazyops.libs.abcs.configs.types import AppEnv
from ..utils.lazy import get_az_settings, logger, get_az_flow, get_az_resource
from typing import Optional, List, Dict, Any, Union
@property
def token_flow(self) -> 'APIClientCredentialsFlow':
    """
        Token Flow
        """
    if self._token_flow is None:
        self._token_flow = get_az_flow(name='api_client_credentials', endpoint=self.endpoint, api_client_id=self.api_client_id, api_client_env=self.api_client_env, audience=self.audience, client_id=self.client_id, client_secret=self.client_secret, oauth_url=self.oauth_url)
    return self._token_flow