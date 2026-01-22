from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
def _inherit_option(self, name: str, val: _T) -> _T:
    """Return the inherited TransactionOption value."""
    if val:
        return val
    txn_opts = self.options.default_transaction_options
    parent_val = txn_opts and getattr(txn_opts, name)
    if parent_val:
        return parent_val
    return getattr(self.client, name)