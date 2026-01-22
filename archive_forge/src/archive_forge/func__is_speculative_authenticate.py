from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def _is_speculative_authenticate(command_name: str, doc: Mapping[str, Any]) -> bool:
    if command_name.lower() in ('hello', HelloCompat.LEGACY_CMD) and 'speculativeAuthenticate' in doc:
        return True
    return False