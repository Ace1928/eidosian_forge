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
def _update_read_concern(self, cmd: MutableMapping[str, Any], conn: Connection) -> None:
    if self.options.causal_consistency and self.operation_time is not None:
        cmd.setdefault('readConcern', {})['afterClusterTime'] = self.operation_time
    if self.options.snapshot:
        if conn.max_wire_version < 13:
            raise ConfigurationError('Snapshot reads require MongoDB 5.0 or later')
        rc = cmd.setdefault('readConcern', {})
        rc['level'] = 'snapshot'
        if self._snapshot_time is not None:
            rc['atClusterTime'] = self._snapshot_time