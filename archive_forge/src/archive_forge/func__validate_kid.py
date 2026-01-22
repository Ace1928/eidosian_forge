from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
def _validate_kid(self, kid: Any) -> None:
    if not isinstance(kid, str):
        raise InvalidTokenError('Key ID header parameter must be a string')