from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def validate_request_token_response(self, resp: Mapping[str, Any]) -> None:
    if not isinstance(resp, dict):
        raise ValueError('OIDC callback returned invalid result')
    if 'access_token' not in resp:
        raise ValueError('OIDC callback did not return an access_token')
    expected = ['access_token', 'refresh_token', 'expires_in_seconds']
    for key in resp:
        if key not in expected:
            raise ValueError(f'Unexpected field in callback result "{key}"')
    self.access_token = resp['access_token']
    self.refresh_token = resp.get('refresh_token')