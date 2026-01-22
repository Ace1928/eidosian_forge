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
def _validate_iss(self, payload: dict[str, Any], issuer: Any) -> None:
    if issuer is None:
        return
    if 'iss' not in payload:
        raise MissingRequiredClaimError('iss')
    if payload['iss'] != issuer:
        raise InvalidIssuerError('Invalid issuer')