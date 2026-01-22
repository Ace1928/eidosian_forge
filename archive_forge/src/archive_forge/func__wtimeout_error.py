from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
def _wtimeout_error(error: Any) -> bool:
    """Return True if this writeConcernError doc is a caused by a timeout."""
    return error.get('code') == 50 or ('errInfo' in error and error['errInfo'].get('wtimeout'))