from __future__ import annotations
import json
import warnings
from calendar import timegm
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from . import api_jws
from .exceptions import (
from .warnings import RemovedInPyjwt3Warning
def _validate_claims(self, payload: dict[str, Any], options: dict[str, Any], audience=None, issuer=None, leeway: float | timedelta=0) -> None:
    if isinstance(leeway, timedelta):
        leeway = leeway.total_seconds()
    if audience is not None and (not isinstance(audience, (str, Iterable))):
        raise TypeError('audience must be a string, iterable or None')
    self._validate_required_claims(payload, options)
    now = datetime.now(tz=timezone.utc).timestamp()
    if 'iat' in payload and options['verify_iat']:
        self._validate_iat(payload, now, leeway)
    if 'nbf' in payload and options['verify_nbf']:
        self._validate_nbf(payload, now, leeway)
    if 'exp' in payload and options['verify_exp']:
        self._validate_exp(payload, now, leeway)
    if options['verify_iss']:
        self._validate_iss(payload, issuer)
    if options['verify_aud']:
        self._validate_aud(payload, audience, strict=options.get('strict_aud', False))