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
def _validate_required_claims(self, payload: dict[str, Any], options: dict[str, Any]) -> None:
    for claim in options['require']:
        if payload.get(claim) is None:
            raise MissingRequiredClaimError(claim)